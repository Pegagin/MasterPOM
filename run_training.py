"""Evolve priority heuristics with GP on disrupted executions of baseline plans.

Usage (from the project root, with the venv):
    python run_training.py                                   # J30, Chand defaults
    python run_training.py --population-size 32 --limit-projects 6   # quick try
    python run_training.py --evolution-basis disruption --preemption-mode restart
    python run_training.py --sgs serial --extra-terminals --output heuristics/serial.json

The best heuristics (ranked on the validation scenarios) are written to a JSON file that
run_evaluation.py reads.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rcpsp_gp.dataset import load_projects, make_scenarios
from rcpsp_gp.gp import save_heuristics
from rcpsp_gp.simulator import PAUSE_ORDERS, SCORE_TYPES, SimulationSettings
from rcpsp_gp.sgs import SCHEDULE_GENERATION_SCHEMES
from rcpsp_gp.split import SPLIT_SCHEMES
from rcpsp_gp.state import PREEMPTION_MODES
from rcpsp_gp.training import EVOLUTION_BASES, TrainingSettings, train_heuristics

DEFAULT_DATA = Path(__file__).parent / "tests" / "data" / "j30"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--data", default=str(DEFAULT_DATA), help="folder with PSPLIB .sm files")
    data.add_argument("--split", choices=SPLIT_SCHEMES, default="chand")
    data.add_argument("--all-instances", action="store_true",
                      help="do not restrict to resource-constrained instances (RS<=0.5, RF>0.5)")
    data.add_argument("--baseline-folder", default="baselines")
    data.add_argument("--replications", type=int, default=3,
                      help="disruption scenarios per instance (Chand: 3)")
    data.add_argument("--limit-projects", type=int, default=None,
                      help="use only the first N training (and validation) instances")

    simulation = parser.add_argument_group("simulation")
    simulation.add_argument("--preemption-mode", choices=PREEMPTION_MODES, default="resume")
    simulation.add_argument("--sgs", choices=list(SCHEDULE_GENERATION_SCHEMES), default="parallel")
    simulation.add_argument("--pause-order", choices=PAUSE_ORDERS, default="baseline_order")
    simulation.add_argument("--allow-early-start", action="store_true",
                            help="let repaired activities start before their baseline day")
    simulation.add_argument("--score", choices=SCORE_TYPES, default="makespan_increase")

    evolution = parser.add_argument_group("genetic programming")
    evolution.add_argument("--population-size", type=int, default=64)
    evolution.add_argument("--evolution-basis", choices=EVOLUTION_BASES, default="project")
    evolution.add_argument("--epochs", type=int, default=1,
                           help="passes over the training scenarios")
    evolution.add_argument("--extra-terminals", action="store_true",
                           help="add the baseline/disruption-aware terminals")
    evolution.add_argument("--top-k", type=int, default=5)
    evolution.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="JSON file for the heuristics (default: heuristics/<settings>.json)")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    simulation_settings = SimulationSettings(
        preemption_mode=arguments.preemption_mode, sgs=arguments.sgs,
        pause_order=arguments.pause_order, no_early_start=not arguments.allow_early_start,
        score_type=arguments.score)
    training_settings = TrainingSettings(
        population_size=arguments.population_size, evolution_basis=arguments.evolution_basis,
        epochs=arguments.epochs, use_extra_terminals=arguments.extra_terminals,
        top_k=arguments.top_k, seed=arguments.seed)

    project_options = dict(split_scheme=arguments.split,
                           resource_constrained_only=not arguments.all_instances,
                           baseline_folder=arguments.baseline_folder,
                           limit=arguments.limit_projects)
    print("loading training and validation projects ...")
    training_projects = load_projects(arguments.data, "train", **project_options)
    validation_projects = load_projects(arguments.data, "validation", **project_options)
    training_scenarios = make_scenarios(training_projects, arguments.replications,
                                        seed=arguments.seed)
    validation_scenarios = make_scenarios(validation_projects, arguments.replications,
                                          seed=arguments.seed + 1)
    print(f"{len(training_projects)} training projects ({len(training_scenarios)} scenarios), "
          f"{len(validation_projects)} validation projects ({len(validation_scenarios)} scenarios)")

    result = train_heuristics(training_scenarios, validation_scenarios, simulation_settings,
                              training_settings)

    output = arguments.output or (f"heuristics/{Path(arguments.data).name}_"
                                  f"{arguments.preemption_mode}_{arguments.sgs}_"
                                  f"{arguments.evolution_basis}_seed{arguments.seed}.json")
    metadata = dict(result.settings, data=str(arguments.data), split=arguments.split,
                    resource_constrained_only=not arguments.all_instances,
                    replications=arguments.replications, history=result.history)
    save_heuristics(output, result.heuristics, result.validation_scores, metadata)

    print("\nbest heuristics (mean validation score, lower is better):")
    for rank, (heuristic, score) in enumerate(zip(result.heuristics, result.validation_scores),
                                              start=1):
        print(f"  GP#{rank}  score={score:7.3f}  size={heuristic.size:>2}  {heuristic.expression}")
    print(f"\n{len(result.history)} generations bred; saved to {output}")


if __name__ == "__main__":
    main()
