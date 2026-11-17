"""Disruption generator, schedule generation schemes and the day-by-day executor."""
import numpy as np
import pytest

from rcpsp_gp.baseline import random_sampling_baseline
from rcpsp_gp.disruption import Disruption, DisruptionGenerator, DisruptionSettings
from rcpsp_gp.generate import random_instance
from rcpsp_gp.metrics import check_feasible, makespan
from rcpsp_gp.rules import STANDARD_RULES
from rcpsp_gp.sgs import schedule_from_scratch
from rcpsp_gp.simulator import SimulationSettings, execute_project
from rcpsp_gp.state import ProjectState


def make_project(seed, n=15, n_res=3):
    instance = random_instance(n=n, n_res=n_res, seed=seed)
    return instance, random_sampling_baseline(instance, n_samples=20, seed=seed)


# ------------------------------------------------------------------ disruption generator
def test_disruption_stream_is_reproducible_and_order_independent():
    generator = DisruptionGenerator(4, DisruptionSettings(0.3, 2.0), seed=7)
    forward = [generator.disruptions_discovered_on(day) for day in range(60)]
    backward = [generator.disruptions_discovered_on(day) for day in reversed(range(60))][::-1]
    assert forward == backward
    assert sum(len(found) for found in forward) > 0


def test_disruption_properties():
    generator = DisruptionGenerator(4, DisruptionSettings(0.5, 2.0), seed=1)
    found = [d for day in range(400) for d in generator.disruptions_discovered_on(day)]
    assert 150 < len(found) < 250                                    # about 50% of days
    for disruption in found:
        assert disruption.start_day == disruption.discovery_day       # no lead time by default
        assert disruption.duration >= 1
        assert 1 <= len(disruption.resources) <= 4
        assert len(set(disruption.resources)) == len(disruption.resources)
        assert all(0 <= f < 1 for f in disruption.reduction_fractions)


def test_no_disruptions_when_failure_probability_is_zero():
    generator = DisruptionGenerator(3, DisruptionSettings(0.0, 2.0), seed=1)
    assert all(not generator.disruptions_discovered_on(day) for day in range(500))


def test_lead_time_delays_the_start():
    generator = DisruptionGenerator(3, DisruptionSettings(1.0, 2.0, mean_lead_time=6), seed=2)
    lead_times = [d.start_day - d.discovery_day for day in range(300)
                  for d in generator.disruptions_discovered_on(day)]
    assert min(lead_times) >= 0 and 4 < np.mean(lead_times) < 7


def test_overlapping_disruptions_compound_and_round_down():
    instance, baseline = make_project(0)
    state = ProjectState(instance, baseline)
    state.apply_disruption(Disruption(0, 2, 3, (0,), (0.5,)))
    state.apply_disruption(Disruption(0, 3, 3, (0,), (0.5,)))
    capacity = instance.capacities[0]
    assert state.available_capacity[2, 0] == np.floor(capacity * 0.5 + 1e-9)
    assert state.available_capacity[3, 0] == np.floor(capacity * 0.25 + 1e-9)
    assert state.available_capacity[5, 0] == np.floor(capacity * 0.5 + 1e-9)
    assert state.availability_profile().window(6, 1)[0, 0] == capacity   # after both ended


# ------------------------------------------------------------------ schedule generation schemes
@pytest.mark.parametrize("sgs", ["parallel", "serial"])
@pytest.mark.parametrize("rule_name", list(STANDARD_RULES))
@pytest.mark.parametrize("seed", range(4))
def test_every_rule_builds_feasible_schedules(sgs, rule_name, seed):
    instance = random_instance(n=15, n_res=3, seed=100 + seed)
    start = schedule_from_scratch(instance, STANDARD_RULES[rule_name], sgs)
    assert check_feasible(instance, start) == []
    assert makespan(instance, start) >= instance.lower_bound


@pytest.mark.parametrize("seed", range(5))
def test_parallel_sgs_ordered_by_a_parallel_schedule_reproduces_it(seed):
    instance = random_instance(n=20, n_res=3, seed=seed)
    start = schedule_from_scratch(instance, STANDARD_RULES["LFT"], "parallel")

    def order_of_that_schedule(view):
        return start[view.candidates].astype(float)

    assert np.array_equal(schedule_from_scratch(instance, order_of_that_schedule, "parallel"),
                          start)


# ------------------------------------------------------------------ executor
@pytest.mark.parametrize("sgs", ["parallel", "serial"])
@pytest.mark.parametrize("seed", range(10))
def test_without_disruptions_the_baseline_is_executed_exactly(sgs, seed):
    instance, baseline = make_project(seed)
    for rule in (STANDARD_RULES["LFT"], STANDARD_RULES["WCS"]):
        result = execute_project(instance, baseline, None, [rule], SimulationSettings(sgs=sgs))
        assert np.array_equal(result.start_day, baseline)
        assert result.makespan == makespan(instance, baseline) and not result.repairs


SETTING_COMBINATIONS = [SimulationSettings(preemption_mode=mode, sgs=sgs, pause_order=order)
                        for mode in ("resume", "restart") for sgs in ("parallel", "serial")
                        for order in ("baseline_order", "rule")]


@pytest.mark.parametrize("settings", SETTING_COMBINATIONS, ids=str)
@pytest.mark.parametrize("seed", range(6))
def test_disrupted_runs_respect_precedence_and_availability(settings, seed):
    instance, baseline = make_project(seed)
    generator = DisruptionGenerator(instance.n_resources, DisruptionSettings(0.25, 2.0), seed)
    rules = [STANDARD_RULES["LFT"], STANDARD_RULES["SPT"], STANDARD_RULES["WCS"]]
    result = execute_project(instance, baseline, generator, rules, settings)
    assert (result.executed_usage <= result.available_capacity).all()
    for activity in range(instance.n_activities):
        for successor in instance.successors[activity]:
            assert result.start_day[successor] >= result.finish_day[activity]
    assert (result.start_day >= baseline).all()                 # no early start
    assert result.makespan >= makespan(instance, baseline)
    if settings.preemption_mode == "restart":     # one contiguous final run per activity
        assert check_feasible(instance, result.start_day, result.available_capacity) == []
        assert (result.finish_day - result.start_day == instance.durations).all()
    else:
        assert result.wasted_work == 0


def test_pool_adopts_the_best_what_if_plan():
    instance, baseline = make_project(3, n=20)
    generator = DisruptionGenerator(instance.n_resources, DisruptionSettings(0.25, 2.0), 3)
    rules = [STANDARD_RULES[name] for name in ("SPT", "LFT", "WCS", "GRD")]
    result = execute_project(instance, baseline, generator, rules)
    assert result.repairs
    for repair in result.repairs:
        assert repair.heuristic_scores[repair.chosen_heuristic] == repair.heuristic_scores.min()
        assert repair.adopted_makespan - makespan(instance, baseline) == \
            repair.heuristic_scores.min()


def test_restart_loses_work_and_resume_does_not():
    instance, baseline = make_project(1, n=20)
    total_wasted = {"resume": 0, "restart": 0}
    for seed in range(10):
        for mode in total_wasted:
            generator = DisruptionGenerator(instance.n_resources, DisruptionSettings(0.3, 2.0),
                                            seed)
            result = execute_project(instance, baseline, generator, [STANDARD_RULES["LFT"]],
                                     SimulationSettings(preemption_mode=mode))
            total_wasted[mode] += result.wasted_work
    assert total_wasted["resume"] == 0 and total_wasted["restart"] > 0


def test_project_without_baseline_builds_its_own_plan():
    instance = random_instance(n=15, n_res=3, seed=9)
    result = execute_project(instance, None, None, [STANDARD_RULES["LFT"]],
                             SimulationSettings(no_early_start=False))
    expected = schedule_from_scratch(instance, STANDARD_RULES["LFT"], "parallel")
    assert np.array_equal(result.start_day, expected)
    assert result.repairs[0].is_initial_plan and not result.disruption_repairs
