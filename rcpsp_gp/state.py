"""Mutable state of a project while it is being executed day by day."""
from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np

from .capacity import CapacityProfile, pad_days
from .disruption import Disruption
from .instance import Instance

PREEMPTION_MODES = ("resume", "restart")


class ActivityStatus(IntEnum):
    WAITING = 0      # not started yet, or paused by a disruption and waiting to continue
    RUNNING = 1
    FINISHED = 2


class ProjectState:
    """Everything known about a running project on the current day.

    Day-indexed tables (``available_capacity``, ``resource_usage``) start at day 0 and grow
    when needed. ``resource_usage`` holds the usage of activities that already ran or are
    running, so its past rows are the executed usage.

    The *plan* (``planned_start`` / ``planned_finish``) covers every activity: actual days
    for started ones, planned days for waiting ones.
    """

    def __init__(self, instance: Instance, baseline_start: Optional[np.ndarray] = None):
        n_activities, n_resources = instance.n_activities, instance.n_resources
        self.instance = instance
        self.baseline_start = None if baseline_start is None else np.asarray(baseline_start)
        self.capacity_factor = np.ones((1, n_resources))          # product of (1 - lost fraction)
        self.available_capacity = instance.capacities[None, :].copy()
        self.resource_usage = np.zeros((1, n_resources), dtype=np.int64)

        self.status = np.full(n_activities, ActivityStatus.WAITING, dtype=np.int64)
        self.start_day = np.full(n_activities, -1, dtype=np.int64)     # realised start (see below)
        self.first_start_day = np.full(n_activities, -1, dtype=np.int64)
        self.segment_start_day = np.full(n_activities, -1, dtype=np.int64)  # start of current run
        self.finish_day = np.full(n_activities, -1, dtype=np.int64)    # known for running/finished
        self.remaining_duration = instance.durations.astype(np.int64).copy()
        self.was_interrupted = np.zeros(n_activities, dtype=bool)
        self.interruption_count = 0
        self.wasted_work = 0           # activity-days lost to restarts

        if self.baseline_start is not None:
            self.planned_start = self.baseline_start.astype(np.int64).copy()
        else:
            self.planned_start = np.full(n_activities, -1, dtype=np.int64)
        self.planned_finish = np.where(self.planned_start >= 0,
                                       self.planned_start + instance.durations, -1)

    # ------------------------------------------------------------------ helpers
    def copy(self) -> "ProjectState":
        clone = ProjectState.__new__(ProjectState)
        for name, value in self.__dict__.items():
            setattr(clone, name, value.copy() if isinstance(value, np.ndarray) else value)
        return clone

    def ensure_days(self, n_days: int) -> None:
        capacities = self.instance.capacities
        self.capacity_factor = pad_days(self.capacity_factor, n_days, np.ones(len(capacities)))
        self.available_capacity = pad_days(self.available_capacity, n_days, capacities)
        self.resource_usage = pad_days(self.resource_usage, n_days, np.zeros_like(capacities))

    def availability_profile(self) -> CapacityProfile:
        """Known available units per day (days beyond what is known: nominal capacity)."""
        return CapacityProfile(self.available_capacity, self.instance.capacities)

    def free_capacity_profile(self) -> CapacityProfile:
        """Units left after the remaining usage of running activities."""
        return CapacityProfile(self.available_capacity - self.resource_usage,
                               self.instance.capacities)

    def activities_with_status(self, status: ActivityStatus) -> np.ndarray:
        return np.flatnonzero(self.status == status)

    def all_finished(self) -> bool:
        return bool(np.all(self.status == ActivityStatus.FINISHED))

    @property
    def planned_makespan(self) -> int:
        return int(self.planned_finish[self.instance.sink])

    # ------------------------------------------------------------------ events
    def apply_disruption(self, disruption: Disruption) -> None:
        self.ensure_days(disruption.end_day)
        days = slice(disruption.start_day, disruption.end_day)
        for resource, lost_fraction in zip(disruption.resources, disruption.reduction_fractions):
            self.capacity_factor[days, resource] *= 1.0 - lost_fraction
            self.available_capacity[days, resource] = np.floor(
                self.instance.capacities[resource] * self.capacity_factor[days, resource] + 1e-9)

    def finish_activities_due(self, day: int) -> None:
        due = (self.status == ActivityStatus.RUNNING) & (self.finish_day <= day)
        self.status[due] = ActivityStatus.FINISHED

    def has_capacity_violation(self, day: int) -> bool:
        self.ensure_days(day + 1)
        return bool(np.any(self.resource_usage[day] > self.available_capacity[day]))

    def start_activity(self, activity: int, day: int) -> None:
        duration = int(self.remaining_duration[activity])
        requirement = self.instance.requirements[activity]
        self.ensure_days(day + duration + 1)
        window = slice(day, day + duration)
        if np.any(self.resource_usage[window] + requirement > self.available_capacity[window]):
            raise RuntimeError(f"plan is infeasible: activity {activity} does not fit on day {day}")
        self.resource_usage[window] += requirement
        if self.start_day[activity] < 0:
            self.start_day[activity] = day
        if self.first_start_day[activity] < 0:
            self.first_start_day[activity] = day
        self.segment_start_day[activity] = day
        self.finish_day[activity] = day + duration
        self.planned_start[activity] = day
        self.planned_finish[activity] = day + duration
        self.status[activity] = (ActivityStatus.FINISHED if duration == 0
                                 else ActivityStatus.RUNNING)

    def pause_activity(self, activity: int, day: int, preemption_mode: str) -> None:
        """Interrupt a running activity on ``day``.

        ``resume`` keeps the finished work (the activity is split) and ``start_day`` stays the
        first start. ``restart`` discards it; ``start_day`` becomes the start of the final run.
        """
        if preemption_mode not in PREEMPTION_MODES:
            raise ValueError(f"preemption_mode must be one of {PREEMPTION_MODES}")
        self.resource_usage[day:self.finish_day[activity]] -= self.instance.requirements[activity]
        if preemption_mode == "resume":
            self.remaining_duration[activity] = self.finish_day[activity] - day
        else:
            self.wasted_work += day - int(self.segment_start_day[activity])
            self.remaining_duration[activity] = self.instance.durations[activity]
            self.start_day[activity] = -1
        self.finish_day[activity] = -1
        self.status[activity] = ActivityStatus.WAITING
        self.was_interrupted[activity] = True
        self.interruption_count += 1

    def adopt_plan(self, planned_start: np.ndarray, planned_finish: np.ndarray) -> None:
        waiting = self.status == ActivityStatus.WAITING
        self.planned_start[waiting] = planned_start[waiting]
        self.planned_finish[waiting] = planned_finish[waiting]
