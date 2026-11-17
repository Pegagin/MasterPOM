"""Day-by-day execution of a baseline plan under disruptions, with heuristic repairs.

How one project run works
-------------------------
1. The plan starts as the baseline (or, without a baseline, is built on day 0).
2. Each day:
   a. running activities whose finish day has come are marked finished;
   b. disruptions discovered today lower the known availability;
   c. if running activities no longer fit today, some are paused (see ``pause_order``);
   d. if anything changed (b or c), every heuristic of the pool plans the rest of the
      project ("what-if" repair). Each repair gets a score and the best plan is adopted;
   e. activities planned to start today are started.
3. The run ends when every activity is finished.

Without disruptions no repair is ever made, so the baseline is executed exactly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Union

import numpy as np

from .decision import DecisionView
from .disruption import Disruption, DisruptionGenerator
from .sgs import SCHEDULE_GENERATION_SCHEMES, Plan
from .state import PREEMPTION_MODES, ActivityStatus, ProjectState

PAUSE_ORDERS = ("baseline_order", "rule")
SCORE_TYPES = ("makespan_increase", "makespan")


@dataclass(frozen=True)
class SimulationSettings:
    """How a project is executed and repaired.

    preemption_mode   ``resume`` keeps finished work of a paused activity, ``restart`` loses it.
    sgs               ``parallel`` or ``serial`` schedule generation scheme for repairs.
    pause_order       which running activities keep running when capacity drops:
                      ``baseline_order`` keeps the ones planned to start earliest in the
                      baseline (Chand's "initial priority"); ``rule`` keeps the ones the
                      repairing heuristic ranks best.
    no_early_start    activities never start before their baseline start day.
    score_type        what-if score of a repair (lower is better): ``makespan_increase`` =
                      planned makespan - baseline makespan, or plain ``makespan``.
    """
    preemption_mode: str = "resume"
    sgs: str = "parallel"
    pause_order: str = "baseline_order"
    no_early_start: bool = True
    score_type: str = "makespan_increase"

    def __post_init__(self):
        for value, allowed, label in ((self.preemption_mode, PREEMPTION_MODES, "preemption_mode"),
                                      (self.sgs, tuple(SCHEDULE_GENERATION_SCHEMES), "sgs"),
                                      (self.pause_order, PAUSE_ORDERS, "pause_order"),
                                      (self.score_type, SCORE_TYPES, "score_type")):
            if value not in allowed:
                raise ValueError(f"{label} must be one of {allowed}, got {value!r}")


# ------------------------------------------------------------------------------ heuristic pools
class HeuristicPool(Protocol):
    """Heuristics consulted at every repair. The GP trainer implements its own pool."""

    def current_heuristics(self) -> Sequence:
        ...

    def record_repair(self, scores: np.ndarray) -> None:
        """Called after each repair with the what-if score of every heuristic."""
        ...


class FixedHeuristicPool:
    """A pool that never changes (one heuristic, or a frozen set of evolved ones)."""

    def __init__(self, heuristics: Sequence):
        if len(heuristics) == 0:
            raise ValueError("a heuristic pool needs at least one heuristic")
        self.heuristics = list(heuristics)

    def current_heuristics(self) -> Sequence:
        return self.heuristics

    def record_repair(self, scores: np.ndarray) -> None:
        pass


# ------------------------------------------------------------------------------ results
@dataclass
class RepairRecord:
    day: int
    new_disruption_count: int
    paused_activities: List[int]
    heuristic_scores: np.ndarray        # what-if score of every heuristic in the pool
    chosen_heuristic: int               # index of the adopted heuristic
    adopted_makespan: int               # planned makespan right after the repair
    runtime_seconds: float
    is_initial_plan: bool = False       # True for the day-0 plan of a project without baseline


@dataclass
class ExecutionResult:
    start_day: np.ndarray               # realised start of every activity
    finish_day: np.ndarray
    makespan: int
    baseline_makespan: Optional[int]
    repairs: List[RepairRecord]
    disruptions: List[Disruption]       # disruptions discovered while the project ran
    interruption_count: int
    wasted_work: int                    # activity-days lost to restarts
    available_capacity: np.ndarray      # [makespan, n_resources] availability as it turned out
    executed_usage: np.ndarray          # [makespan, n_resources] units actually used
    capacities: np.ndarray = field(repr=False, default=None)

    @property
    def disruption_repairs(self) -> List[RepairRecord]:
        return [repair for repair in self.repairs if not repair.is_initial_plan]


# ------------------------------------------------------------------------------ execution
def what_if_score(planned_makespan: int, baseline_makespan: Optional[int],
                  score_type: str) -> float:
    if score_type == "makespan_increase" and baseline_makespan is not None:
        return float(planned_makespan - baseline_makespan)
    return float(planned_makespan)


def pause_over_capacity(state: ProjectState, day: int, pause_order: str, preemption_mode: str,
                        rule=None) -> List[int]:
    """Pause running activities until the rest fits the availability of ``day``.

    Running activities are re-admitted one by one in priority order (Chand et al. 2019,
    Section 3); an activity that no longer fits next to the already re-admitted ones is
    paused. Returns the paused activity ids.
    """
    running = state.activities_with_status(ActivityStatus.RUNNING)
    if len(running) == 0:
        return []
    instance = state.instance
    if pause_order == "baseline_order":
        initial_order = (state.baseline_start if state.baseline_start is not None
                         else state.first_start_day)
        priorities = initial_order[running].astype(float)
    else:
        view = DecisionView(state, running, day, state.free_capacity_profile(),
                            remaining_durations=state.finish_day[running] - day)
        with np.errstate(all="ignore"):
            priorities = np.nan_to_num(np.asarray(rule(view), dtype=float), nan=np.inf)
    readmitted_usage = np.zeros(instance.n_resources, dtype=np.int64)
    available_today = state.available_capacity[day]
    paused = []
    for activity in running[np.lexsort((running, priorities))]:
        requirement = instance.requirements[activity]
        if np.all(readmitted_usage + requirement <= available_today):
            readmitted_usage += requirement
        else:
            state.pause_activity(int(activity), day, preemption_mode)
            paused.append(int(activity))
    return paused


def _repair(state: ProjectState, day: int, heuristics: Sequence, settings: SimulationSettings,
            baseline_makespan: Optional[int], over_capacity: bool):
    """Let every heuristic plan the rest of the project.

    Returns ``(state with the best plan, score of every heuristic, index of the best,
    activities paused today)``.
    """
    plan_rest = SCHEDULE_GENERATION_SCHEMES[settings.sgs]
    paused_by_baseline_order: List[int] = []
    if over_capacity and settings.pause_order == "baseline_order":
        paused_by_baseline_order = pause_over_capacity(state, day, "baseline_order",
                                                       settings.preemption_mode)
    candidate_states, scores, paused_lists = [], [], []
    for heuristic in heuristics:
        candidate = state.copy() if len(heuristics) > 1 else state
        paused = paused_by_baseline_order
        if over_capacity and settings.pause_order == "rule":
            paused = pause_over_capacity(candidate, day, "rule", settings.preemption_mode,
                                         heuristic)
        plan: Plan = plan_rest(candidate, heuristic, day, settings.no_early_start)
        candidate.adopt_plan(plan.start, plan.finish)
        candidate_states.append(candidate)
        scores.append(what_if_score(plan.makespan, baseline_makespan, settings.score_type))
        paused_lists.append(paused)
    scores_array = np.array(scores)
    best = int(np.argmin(scores_array))          # the first heuristic wins ties
    return candidate_states[best], scores_array, best, paused_lists[best]


def execute_project(instance, baseline_start: Optional[np.ndarray],
                    disruption_generator: Optional[DisruptionGenerator],
                    heuristics: Union[HeuristicPool, Sequence],
                    settings: SimulationSettings = SimulationSettings(),
                    max_days: Optional[int] = None) -> ExecutionResult:
    """Execute one project day by day; see the module docstring for the steps.

    ``heuristics`` is either a :class:`HeuristicPool` or a plain list of priority rules.
    """
    pool = heuristics if hasattr(heuristics, "record_repair") else FixedHeuristicPool(heuristics)
    state = ProjectState(instance, baseline_start)
    baseline_makespan = None if baseline_start is None else int(baseline_start[instance.sink])
    if max_days is None:
        max_days = 50 * (int(instance.durations.sum()) + 1) + 1000
    repairs: List[RepairRecord] = []
    discovered: List[Disruption] = []

    if baseline_start is None:              # build the initial plan with the heuristics
        started = time.perf_counter()
        state, scores, best, _ = _repair(state, 0, pool.current_heuristics(), settings,
                                         None, over_capacity=False)
        repairs.append(RepairRecord(0, 0, [], scores, best, state.planned_makespan,
                                    time.perf_counter() - started, is_initial_plan=True))

    day = 0
    while True:
        state.finish_activities_due(day)
        if state.all_finished():
            break
        if day > max_days:
            raise RuntimeError(f"project {instance.name} did not finish within {max_days} days")
        new_disruptions = (disruption_generator.disruptions_discovered_on(day)
                           if disruption_generator is not None else [])
        for disruption in new_disruptions:
            state.apply_disruption(disruption)
        discovered.extend(new_disruptions)
        over_capacity = state.has_capacity_violation(day)
        if new_disruptions or over_capacity:
            started = time.perf_counter()
            state, scores, best, paused = _repair(state, day, pool.current_heuristics(),
                                                  settings, baseline_makespan, over_capacity)
            repairs.append(RepairRecord(day, len(new_disruptions), paused, scores, best,
                                        state.planned_makespan, time.perf_counter() - started))
            pool.record_repair(scores)
        starting_today = [int(a) for a in state.activities_with_status(ActivityStatus.WAITING)
                          if state.planned_start[a] == day]
        for activity in starting_today:
            state.start_activity(activity, day)
        day += 1

    makespan = int(state.finish_day[instance.sink])
    state.ensure_days(makespan + 1)
    return ExecutionResult(
        start_day=state.start_day.copy(), finish_day=state.finish_day.copy(), makespan=makespan,
        baseline_makespan=baseline_makespan, repairs=repairs, disruptions=discovered,
        interruption_count=state.interruption_count, wasted_work=state.wasted_work,
        available_capacity=state.available_capacity[:makespan].copy(),
        executed_usage=state.resource_usage[:makespan].copy(),
        capacities=instance.capacities.copy())


def run_scenario(scenario, heuristics: Union[HeuristicPool, Sequence],
                 settings: SimulationSettings = SimulationSettings()) -> ExecutionResult:
    """Execute a :class:`~rcpsp_gp.dataset.Scenario` (project + disruption stream)."""
    return execute_project(scenario.project.instance, scenario.project.baseline_start,
                           scenario.disruption_generator(), heuristics, settings)
