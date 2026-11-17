"""Online training of a population of priority heuristics with genetic programming.

During training the whole population is the heuristic pool of the simulator: at every
disruption each heuristic plans a what-if repair, the best plan is adopted, and every
heuristic's score is recorded. When the population breeds depends on the evolution basis:

``project`` (default)
    The population is fixed for a whole project run. When the project ends, each
    heuristic's fitness is its mean score over that project's repairs and the population
    breeds once. The new population continues with the next project.
``disruption``
    The population breeds right after every repair, using that repair's scores, so the
    next disruption is already handled by the new generation.

Breeding: tournament selection, one-point crossover, uniform subtree mutation, a tree depth
limit and a few elites copied unchanged (fitness is minimised).

After training, the final population and the best heuristic of every generation are run
one at a time on the validation scenarios and ranked by their mean realised score; the best
``top_k`` are returned (Chand et al. 2019 also pick final rules on a validation set).
"""
from __future__ import annotations

import operator
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from deap import base, creator, gp, tools

from .dataset import Scenario
from .gp import EvolvedHeuristic, build_primitive_set, ensure_deap_types, terminal_names
from .simulator import SimulationSettings, run_scenario, what_if_score

EVOLUTION_BASES = ("project", "disruption")


@dataclass(frozen=True)
class TrainingSettings:
    population_size: int = 64
    evolution_basis: str = "project"
    epochs: int = 1                        # passes over the training scenarios
    tournament_size: int = 7
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    max_tree_depth: int = 6
    initial_min_depth: int = 2
    initial_max_depth: int = 4
    elite_count: int = 2                   # best heuristics copied unchanged into the next generation
    use_extra_terminals: bool = False
    top_k: int = 5                         # how many heuristics to keep after validation
    seed: int = 0

    def __post_init__(self):
        if self.evolution_basis not in EVOLUTION_BASES:
            raise ValueError(f"evolution_basis must be one of {EVOLUTION_BASES}")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be smaller than population_size")


class EvolvingHeuristicPool:
    """The GP population, used by the simulator as its heuristic pool."""

    def __init__(self, primitive_set: gp.PrimitiveSet, settings: TrainingSettings):
        ensure_deap_types()
        self.primitive_set = primitive_set
        self.settings = settings
        toolbox = base.Toolbox()
        toolbox.register("expression", gp.genHalfAndHalf, pset=primitive_set,
                         min_=settings.initial_min_depth, max_=settings.initial_max_depth)
        toolbox.register("individual", tools.initIterate, creator.PriorityTree, toolbox.expression)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selTournament, tournsize=settings.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("subtree", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.subtree, pset=primitive_set)
        depth_limit = gp.staticLimit(key=operator.attrgetter("height"),
                                     max_value=settings.max_tree_depth)
        toolbox.decorate("mate", depth_limit)
        toolbox.decorate("mutate", depth_limit)
        self.toolbox = toolbox

        self.population = toolbox.population(n=settings.population_size)
        self.generation = 0
        self.history: List[Dict] = []                         # one entry per breeding step
        self.generation_champions: Dict[str, gp.PrimitiveTree] = {}
        self._compiled: Dict[str, EvolvedHeuristic] = {}
        self._scores_this_project: List[np.ndarray] = []

    # -------------------------------------------------------------- HeuristicPool interface
    def current_heuristics(self) -> List[EvolvedHeuristic]:
        return [self.heuristic(tree) for tree in self.population]

    def record_repair(self, scores: np.ndarray) -> None:
        if self.settings.evolution_basis == "disruption":
            self.breed(scores)
        else:
            self._scores_this_project.append(np.asarray(scores, dtype=float))

    # -------------------------------------------------------------- evolution
    def heuristic(self, tree: gp.PrimitiveTree) -> EvolvedHeuristic:
        expression = str(tree)
        if expression not in self._compiled:
            self._compiled[expression] = EvolvedHeuristic(tree, self.primitive_set)
        return self._compiled[expression]

    def finish_project(self) -> bool:
        """End of a project run. With the project basis the population breeds now.

        Returns whether it bred (a project without disruptions gives no fitness)."""
        bred = False
        if self.settings.evolution_basis == "project" and self._scores_this_project:
            self.breed(np.mean(self._scores_this_project, axis=0))
            bred = True
        self._scores_this_project = []
        return bred

    def breed(self, fitness_values: np.ndarray) -> None:
        settings, toolbox = self.settings, self.toolbox
        for tree, value in zip(self.population, fitness_values):
            tree.fitness.values = (float(value),)
        ranking = np.argsort(fitness_values, kind="stable")
        champion = self.population[ranking[0]]
        self.generation_champions[str(champion)] = toolbox.clone(champion)
        self.history.append({
            "generation": self.generation,
            "best_fitness": float(fitness_values[ranking[0]]),
            "mean_fitness": float(np.mean(fitness_values)),
            "mean_size": float(np.mean([len(tree) for tree in self.population])),
            "best_expression": str(champion),
        })

        elites = [toolbox.clone(self.population[i]) for i in ranking[:settings.elite_count]]
        offspring = [toolbox.clone(tree) for tree in
                     toolbox.select(self.population, settings.population_size - settings.elite_count)]
        for i in range(1, len(offspring), 2):
            if random.random() < settings.crossover_probability:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
        for i in range(len(offspring)):
            if random.random() < settings.mutation_probability:
                offspring[i], = toolbox.mutate(offspring[i])
        for tree in offspring:
            del tree.fitness.values
        self.population = elites + offspring
        self.generation += 1
        alive = {str(tree) for tree in self.population}
        self._compiled = {e: h for e, h in self._compiled.items() if e in alive}


# ------------------------------------------------------------------------------ training run
@dataclass
class TrainingResult:
    heuristics: List[EvolvedHeuristic]          # best first, at most top_k
    validation_scores: List[float]              # mean realised score of each (lower is better)
    history: List[Dict]                         # one entry per breeding step
    settings: Dict = field(default_factory=dict)


def mean_realised_score(heuristic, scenarios: Sequence[Scenario],
                        simulation_settings: SimulationSettings) -> float:
    """Mean score of one heuristic used for every repair (no pool)."""
    scores = []
    for scenario in scenarios:
        result = run_scenario(scenario, [heuristic], simulation_settings)
        scores.append(what_if_score(result.makespan, result.baseline_makespan,
                                    simulation_settings.score_type))
    return float(np.mean(scores))


def train_heuristics(training_scenarios: Sequence[Scenario],
                     validation_scenarios: Sequence[Scenario],
                     simulation_settings: SimulationSettings = SimulationSettings(),
                     training_settings: TrainingSettings = TrainingSettings(),
                     log: Optional[Callable[[str], None]] = print,
                     progress: Optional[Callable[[str, int, int, List[Dict]], None]] = None
                     ) -> TrainingResult:
    """Train, validate and return the best heuristics.

    ``progress(phase, done, total, history)`` is called after every training project
    (phase ``"training"``) and every validated candidate (phase ``"validation"``); the UI
    uses it for progress bars and the live fitness chart.
    """
    log = log or (lambda message: None)
    progress = progress or (lambda phase, done, total, history: None)
    total_projects = training_settings.epochs * len(training_scenarios)
    projects_done = 0
    random.seed(training_settings.seed)
    primitive_set = build_primitive_set(terminal_names(training_settings.use_extra_terminals))
    pool = EvolvingHeuristicPool(primitive_set, training_settings)
    shuffler = random.Random(training_settings.seed)

    started = time.perf_counter()
    for epoch in range(1, training_settings.epochs + 1):
        order = list(training_scenarios)
        shuffler.shuffle(order)
        for index, scenario in enumerate(order, start=1):
            result = run_scenario(scenario, pool, simulation_settings)
            pool.finish_project()
            best = pool.history[-1]["best_fitness"] if pool.history else float("nan")
            log(f"epoch {epoch} project {index:>3}/{len(order)} {scenario.label:<45} "
                f"repairs={len(result.repairs):>3} generation={pool.generation:>4} "
                f"last best fitness={best:7.2f} ({time.perf_counter() - started:6.1f}s)")
            projects_done += 1
            progress("training", projects_done, total_projects, pool.history)

    candidates: Dict[str, gp.PrimitiveTree] = dict(pool.generation_champions)
    for tree in pool.population:
        candidates.setdefault(str(tree), tree)
    if not validation_scenarios:
        log("no validation scenarios given; ranking candidates on the training scenarios")
        validation_scenarios = training_scenarios
    log(f"validating {len(candidates)} candidate heuristics on "
        f"{len(validation_scenarios)} scenarios ...")
    scored = []
    for done, tree in enumerate(candidates.values(), start=1):
        heuristic = EvolvedHeuristic(tree, primitive_set)
        scored.append((mean_realised_score(heuristic, validation_scenarios, simulation_settings),
                       heuristic.size, heuristic.expression, heuristic))
        progress("validation", done, len(candidates), pool.history)
    scored.sort(key=lambda entry: entry[:3])          # score, then smaller tree, then text
    best = scored[:training_settings.top_k]
    log(f"training finished in {time.perf_counter() - started:.1f}s")
    return TrainingResult(heuristics=[entry[3] for entry in best],
                          validation_scores=[entry[0] for entry in best],
                          history=pool.history,
                          settings={"training": asdict(training_settings),
                                    "simulation": asdict(simulation_settings)})
