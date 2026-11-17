"""GP heuristics, training loop and the evaluation harness (small, fast settings)."""
import numpy as np
import pytest

from rcpsp_gp.baseline import random_sampling_baseline
from rcpsp_gp.dataset import Project, make_scenarios
from rcpsp_gp.disruption import DisruptionSettings
from rcpsp_gp.evaluation import (check_baseline_reproduction, compare_methods, evaluate_methods,
                                 evolved_methods, standard_methods, summary_report,
                                 write_rows_csv)
from rcpsp_gp.generate import random_instance
from rcpsp_gp.gp import (EvolvedHeuristic, build_primitive_set, load_heuristics,
                         protected_division, save_heuristics, terminal_names)
from rcpsp_gp.metrics import check_feasible
from rcpsp_gp.rules import STANDARD_RULES
from rcpsp_gp.sgs import schedule_from_scratch
from rcpsp_gp.simulator import SimulationSettings, run_scenario
from rcpsp_gp.training import TrainingSettings, train_heuristics


def make_projects(count, first_seed=0):
    projects = []
    for seed in range(first_seed, first_seed + count):
        instance = random_instance(n=12, n_res=3, seed=seed)
        baseline = random_sampling_baseline(instance, n_samples=20, seed=seed)
        projects.append(Project(instance, baseline, parameter_set=1, instance_number=seed,
                                resource_factor=1.0, resource_strength=0.2))
    return projects


def test_protected_division():
    result = protected_division(np.array([1.0, 2.0, 3.0]), np.array([2.0, 0.0, -1.0]))
    assert list(result) == [0.5, 0.0, -3.0]


@pytest.mark.parametrize("use_extra", [False, True])
def test_every_terminal_can_be_evaluated(use_extra):
    primitive_set = build_primitive_set(terminal_names(use_extra))
    project = make_projects(1)[0]
    for name in terminal_names(use_extra):
        heuristic = EvolvedHeuristic.from_expression(f"add({name}, {name})", primitive_set)
        start = schedule_from_scratch(project.instance, heuristic, "parallel")
        assert check_feasible(project.instance, start) == []


def test_evolved_expression_equals_hand_written_rule():
    """An evolved tree that is just LS must order like the standard LST rule."""
    primitive_set = build_primitive_set(terminal_names())
    latest_start = EvolvedHeuristic.from_expression("LS", primitive_set)
    for project in make_projects(3):
        for sgs in ("parallel", "serial"):
            assert np.array_equal(schedule_from_scratch(project.instance, latest_start, sgs),
                                  schedule_from_scratch(project.instance, STANDARD_RULES["LST"],
                                                        sgs))


def test_save_and_load_heuristics(tmp_path):
    primitive_set = build_primitive_set(terminal_names())
    heuristics = [EvolvedHeuristic.from_expression(text, primitive_set)
                  for text in ("sub(LF, AvgRReq)", "add(LS, mul(TSC, neg(RR)))")]
    path = save_heuristics(tmp_path / "h.json", heuristics, [1.5, 2.0], {"note": "test"})
    loaded, document = load_heuristics(path)
    assert [h.expression for h in loaded] == [h.expression for h in heuristics]
    assert document["metadata"]["note"] == "test"
    assert document["heuristics"][0]["validation_score"] == 1.5


@pytest.mark.parametrize("basis", ["project", "disruption"])
def test_training_runs_and_returns_ranked_heuristics(basis):
    training = make_scenarios(make_projects(3), replications=1, failure_probabilities=(0.25,),
                              mean_repair_times=(2,), seed=0)
    validation = make_scenarios(make_projects(2, first_seed=50), replications=1,
                                failure_probabilities=(0.25,), mean_repair_times=(2,), seed=1)
    settings = TrainingSettings(population_size=8, evolution_basis=basis, top_k=3,
                                tournament_size=3, seed=0)
    result = train_heuristics(training, validation, SimulationSettings(), settings, log=None)
    assert 1 <= len(result.heuristics) <= 3
    assert result.validation_scores == sorted(result.validation_scores)
    assert result.history, "the population never bred"
    if basis == "disruption":
        assert len(result.history) > len(training)       # bred after every repair


def test_evaluation_rows_summary_and_control(tmp_path):
    projects = make_projects(3)
    scenarios = make_scenarios(projects, replications=2, seed=5)
    primitive_set = build_primitive_set(terminal_names())
    evolved = [EvolvedHeuristic.from_expression(text, primitive_set)
               for text in ("sub(LF, AvgRReq)", "LS", "add(LS, TSC)")]
    methods = standard_methods(["LFT", "SPT"]) + evolved_methods(evolved, top_k=2)
    assert [m.name for m in methods] == ["LFT", "SPT", "GP#1", "GP#2", "GP-pool"]

    settings = SimulationSettings()
    assert check_baseline_reproduction(methods, projects, settings) == []
    rows = evaluate_methods(methods, scenarios, settings, log=None)
    assert len(rows) == len(scenarios) * len(methods)

    # all methods meet the first disruption on the same day, in the same situation
    for scenario in scenarios:
        first_days = {row["first_disruption_day"] for row in rows
                      if (row["instance"], row["seed"]) == (scenario.project.name, scenario.seed)}
        assert len(first_days) == 1
    pool_rows = [row for row in rows if row["method"] == "GP-pool"]
    assert all(len(row["pool_choice_counts"].split()) == 2 for row in pool_rows if row["repair_count"])

    comparison = compare_methods(rows, "GP#1", "LFT")
    assert comparison["better"] + comparison["equal"] + comparison["worse"] == len(scenarios)
    report = summary_report(rows)
    assert "Mean per method" in report and "GP-pool" in report
    assert write_rows_csv(rows, tmp_path / "rows.csv").exists()


def test_pool_is_never_worse_than_its_members_at_the_first_repair():
    """At the first repair all methods are in the same state, so the pool's plan must be at
    least as good as each member's plan."""
    projects = make_projects(4)
    scenarios = make_scenarios(projects, replications=2, failure_probabilities=(0.25,), seed=3)
    members = [STANDARD_RULES[name] for name in ("LFT", "SPT", "WCS")]
    for scenario in scenarios:
        pool_result = run_scenario(scenario, members)
        if not pool_result.disruption_repairs:
            continue
        pool_first = pool_result.disruption_repairs[0].adopted_makespan
        for member in members:
            member_first = run_scenario(scenario, [member]).disruption_repairs[0].adopted_makespan
            assert pool_first <= member_first


def test_disruption_settings_of_scenarios_come_from_chand_values():
    scenarios = make_scenarios(make_projects(5), replications=3, seed=0)
    assert len(scenarios) == 15
    assert {s.disruption_settings.failure_probability for s in scenarios} <= {0.15, 0.2, 0.25}
    assert {s.disruption_settings.mean_repair_time for s in scenarios} <= {1, 2, 3}
    assert all(isinstance(s.disruption_settings, DisruptionSettings) for s in scenarios)
