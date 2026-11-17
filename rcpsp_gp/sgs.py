"""Schedule generation schemes (SGS) that plan every waiting activity from a given day on.

Both schemes take the same inputs and return a :class:`Plan`, so the simulator can swap
them with a setting:

``parallel`` (default, as in Chand et al.)
    Moves a clock forward day by day. On each day it keeps starting the eligible activity
    with the best priority until nothing else fits (non-delay scheduling).
``serial``
    Picks one activity at a time among those whose predecessors are all planned and puts it
    at the earliest day it fits. It may leave gaps and fill them later. For the priority
    rule, each candidate's "current day" is its earliest precedence-feasible start day.

Both respect the known availability, the remaining usage of running activities and, with
``no_early_start``, the baseline start days as release dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .decision import DecisionView
from .state import ActivityStatus, ProjectState


@dataclass
class Plan:
    start: np.ndarray       # planned (or actual) start day of every activity
    finish: np.ndarray      # planned (or actual) finish day of every activity

    @property
    def makespan(self) -> int:
        return int(self.finish[-1])         # the sink is the last activity


def pick_by_rule(rule, view: DecisionView) -> int:
    """Candidate with the lowest priority value; ties go to the lower activity id."""
    if view.n_candidates == 1:
        return int(view.candidates[0])
    with np.errstate(all="ignore"):
        priorities = np.asarray(rule(view), dtype=float)
    priorities = np.broadcast_to(priorities, view.candidates.shape)
    priorities = np.where(np.isnan(priorities), np.inf, priorities)
    best = np.lexsort((view.candidates, priorities))[0]
    return int(view.candidates[best])


class _PlanningContext:
    """Bookkeeping shared by both schemes while they build one plan."""

    def __init__(self, state: ProjectState, from_day: int, no_early_start: bool):
        instance = state.instance
        self.state = state
        self.from_day = from_day
        self.free_capacity = state.free_capacity_profile()
        self.planned_start = state.planned_start.copy()
        self.planned_finish = state.planned_finish.copy()
        started = state.status != ActivityStatus.WAITING
        self.known_finish = np.where(started, state.finish_day, -1)   # -1 = not planned yet
        self.waiting: List[int] = sorted(int(a) for a in
                                         state.activities_with_status(ActivityStatus.WAITING))
        if no_early_start and state.baseline_start is not None:
            self.release_day = state.baseline_start.astype(np.int64)
        else:
            self.release_day = np.zeros(instance.n_activities, dtype=np.int64)
        remaining_work = int(state.remaining_duration[self.waiting].sum()) if self.waiting else 0
        self.last_possible_day = (max(from_day, len(state.available_capacity),
                                      int(self.release_day.max())) + remaining_work + 1)

    def predecessors_finished_by(self, activity: int, day: int) -> bool:
        finish = self.known_finish[self.state.instance.predecessors[activity]]
        return bool(np.all((finish >= 0) & (finish <= day)))

    def predecessors_planned(self, activity: int) -> bool:
        return bool(np.all(self.known_finish[self.state.instance.predecessors[activity]] >= 0))

    def earliest_precedence_start(self, activity: int) -> int:
        predecessor_finish = self.known_finish[self.state.instance.predecessors[activity]]
        latest_predecessor = int(predecessor_finish.max()) if len(predecessor_finish) else 0
        return max(self.from_day, int(self.release_day[activity]), latest_predecessor)

    def fits(self, activity: int, day: int) -> bool:
        return self.free_capacity.fits(self.state.instance.requirements[activity], day,
                                       int(self.state.remaining_duration[activity]))

    def place(self, activity: int, day: int) -> None:
        duration = int(self.state.remaining_duration[activity])
        self.free_capacity.reserve(self.state.instance.requirements[activity], day, duration)
        self.planned_start[activity] = day
        self.planned_finish[activity] = day + duration
        self.known_finish[activity] = day + duration
        self.waiting.remove(activity)

    def plan(self) -> Plan:
        return Plan(self.planned_start, self.planned_finish)


def plan_parallel(state: ProjectState, rule, from_day: int, no_early_start: bool = True) -> Plan:
    context = _PlanningContext(state, from_day, no_early_start)
    day = from_day
    while context.waiting:
        if day > context.last_possible_day:
            raise RuntimeError("parallel SGS made no progress; the instance may be infeasible")
        while True:
            ready = [a for a in context.waiting
                     if context.release_day[a] <= day and context.predecessors_finished_by(a, day)]
            fitting = [a for a in ready if context.fits(a, day)]
            if not fitting:
                break
            view = DecisionView(state, fitting, day, context.free_capacity)
            context.place(pick_by_rule(rule, view), day)
        day += 1
    return context.plan()


def plan_serial(state: ProjectState, rule, from_day: int, no_early_start: bool = True) -> Plan:
    context = _PlanningContext(state, from_day, no_early_start)
    instance = state.instance
    while context.waiting:
        eligible = [a for a in context.waiting if context.predecessors_planned(a)]
        start_days = [context.earliest_precedence_start(a) for a in eligible]
        view = DecisionView(state, eligible, start_days, context.free_capacity)
        chosen = pick_by_rule(rule, view)
        earliest = start_days[eligible.index(chosen)]
        day = context.free_capacity.earliest_fit(instance.requirements[chosen], earliest,
                                                 int(state.remaining_duration[chosen]))
        context.place(chosen, day)
    return context.plan()


ScheduleGenerationScheme = Callable[[ProjectState, object, int, bool], Plan]
SCHEDULE_GENERATION_SCHEMES: Dict[str, ScheduleGenerationScheme] = {
    "parallel": plan_parallel,
    "serial": plan_serial,
}


def schedule_from_scratch(instance, rule, sgs: str = "parallel") -> np.ndarray:
    """Start days of a complete schedule built by ``rule`` with nominal capacities."""
    state = ProjectState(instance)
    return SCHEDULE_GENERATION_SCHEMES[sgs](state, rule, 0, False).start
