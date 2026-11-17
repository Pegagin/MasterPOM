"""What a priority rule sees at one scheduling decision, and the GP terminals derived from it.

A priority rule receives a :class:`DecisionView` and returns one priority value per candidate
activity. **Lower value = scheduled first**; ties go to the lower activity id.

Terminals (Chand et al. 2019, Table 4) are normalised to roughly 0..1 so evolved rules do
not depend on the units of the instance:

====================  ============================================================
ES, EF, LS, LF        critical-path times / lower bound
TPC, TSC              number of (direct + indirect) predecessors / successors / (n - 1)
RR                    share of the resource types the activity needs
AvgRReq               mean over resources of requirement / known availability, averaged
                      over the activity's remaining duration from its candidate start day
MaxRReq, MinRReq      the same, maximum / minimum over resources (the paper's Table 4 swaps
                      Max and Min; here the names mean what they say)
====================  ============================================================

Optional extra terminals for the baseline-driven design (off by default):

====================  ============================================================
PlannedStart          baseline start / baseline makespan
Delay                 (candidate start day - baseline start) / baseline makespan
Remaining             remaining duration / longest activity duration
Interrupted           1 if the activity was paused by a disruption before, else 0
TimePressure          (LS - candidate start day) / lower bound
AvailabilityRatio     mean known availability / capacity of the needed resources, on the
                      candidate start day
TimeToRecovery        days until all needed resources are back at nominal capacity
                      / lower bound
====================  ============================================================
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .capacity import CapacityProfile
from .state import ProjectState

CHAND_TERMINALS = ("ES", "EF", "LS", "LF", "TPC", "TSC", "RR", "AvgRReq", "MaxRReq", "MinRReq")
EXTRA_TERMINALS = ("PlannedStart", "Delay", "Remaining", "Interrupted", "TimePressure",
                   "AvailabilityRatio", "TimeToRecovery")


class DecisionView:
    """The candidates of one scheduling decision plus read access to the project state.

    ``candidate_start_days`` is the day each candidate would start: the current day in the
    parallel SGS, the earliest precedence-feasible day in the serial SGS.
    ``free_capacity`` is what is left after running and already planned activities.
    """

    def __init__(self, state: ProjectState, candidates, candidate_start_days,
                 free_capacity: CapacityProfile,
                 remaining_durations: Optional[np.ndarray] = None):
        self.state = state
        self.instance = state.instance
        self.candidates = np.asarray(candidates, dtype=np.int64)
        self.candidate_start_days = np.broadcast_to(
            np.asarray(candidate_start_days, dtype=np.int64), self.candidates.shape)
        self.free_capacity = free_capacity
        self.remaining_durations = (state.remaining_duration[self.candidates]
                                    if remaining_durations is None
                                    else np.asarray(remaining_durations, dtype=np.int64))
        self._availability: Optional[CapacityProfile] = None
        self._terminal_cache: Dict[str, np.ndarray] = {}
        self._requirement_ratios: Optional[np.ndarray] = None

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def availability(self) -> CapacityProfile:
        """Known available units per day (disruptions discovered so far)."""
        if self._availability is None:
            self._availability = self.state.availability_profile()
        return self._availability

    def terminal(self, name: str) -> np.ndarray:
        """Value of terminal ``name`` for every candidate (cached per decision)."""
        if name not in self._terminal_cache:
            self._terminal_cache[name] = np.asarray(TERMINALS[name](self), dtype=float)
        return self._terminal_cache[name]

    def requirement_ratios(self) -> np.ndarray:
        """``[candidate, resource]`` mean of requirement / known availability over the
        candidate's remaining duration. Ratios are capped at 1; a needed resource with zero
        availability counts as 1, an unused resource as 0."""
        if self._requirement_ratios is None:
            ratios = np.zeros((self.n_candidates, self.instance.n_resources))
            for row, activity in enumerate(self.candidates):
                duration = int(self.remaining_durations[row])
                if duration <= 0:
                    continue
                requirement = self.instance.requirements[activity].astype(float)
                window = self.availability.window(int(self.candidate_start_days[row]), duration)
                with np.errstate(divide="ignore", invalid="ignore"):
                    per_day = np.where(window > 0, np.minimum(requirement / window, 1.0), 1.0)
                per_day = np.where(requirement > 0, per_day, 0.0)
                ratios[row] = per_day.mean(axis=0)
            self._requirement_ratios = ratios
        return self._requirement_ratios


# ---------------------------------------------------------------------- terminal functions
def _lower_bound(view: DecisionView) -> float:
    return float(max(view.instance.lower_bound, 1))


def _baseline_makespan(view: DecisionView) -> float:
    baseline = view.state.baseline_start
    if baseline is None:
        return _lower_bound(view)
    return float(max(baseline[view.instance.sink], 1))


def _needed_resources(view: DecisionView) -> np.ndarray:
    return view.instance.requirements[view.candidates] > 0


def _availability_ratio(view: DecisionView) -> np.ndarray:
    capacities = view.instance.capacities
    needed = _needed_resources(view)
    values = np.ones(view.n_candidates)
    for row, start_day in enumerate(view.candidate_start_days):
        if needed[row].any():
            share = view.availability.window(int(start_day), 1)[0] / capacities
            values[row] = share[needed[row]].mean()
    return values


def _time_to_recovery(view: DecisionView) -> np.ndarray:
    capacities = view.instance.capacities
    known_days = len(view.state.available_capacity)
    needed = _needed_resources(view)
    values = np.zeros(view.n_candidates)
    for row, start_day in enumerate(view.candidate_start_days):
        day = int(start_day)
        while day < known_days and np.any(
                view.state.available_capacity[day, needed[row]] < capacities[needed[row]]):
            day += 1
        values[row] = (day - start_day) / _lower_bound(view)
    return values


def _planned_start(view: DecisionView) -> np.ndarray:
    baseline = view.state.baseline_start
    if baseline is None:
        return np.zeros(view.n_candidates)
    return baseline[view.candidates] / _baseline_makespan(view)


def _delay(view: DecisionView) -> np.ndarray:
    baseline = view.state.baseline_start
    if baseline is None:
        return np.zeros(view.n_candidates)
    return (view.candidate_start_days - baseline[view.candidates]) / _baseline_makespan(view)


TERMINALS: Dict[str, Callable[[DecisionView], np.ndarray]] = {
    "ES": lambda v: v.instance.earliest_start[v.candidates] / _lower_bound(v),
    "EF": lambda v: v.instance.earliest_finish[v.candidates] / _lower_bound(v),
    "LS": lambda v: v.instance.latest_start[v.candidates] / _lower_bound(v),
    "LF": lambda v: v.instance.latest_finish[v.candidates] / _lower_bound(v),
    "TPC": lambda v: v.instance.total_predecessor_count[v.candidates]
    / max(v.instance.n_activities - 1, 1),
    "TSC": lambda v: v.instance.total_successor_count[v.candidates]
    / max(v.instance.n_activities - 1, 1),
    "RR": lambda v: _needed_resources(v).mean(axis=1),
    "AvgRReq": lambda v: v.requirement_ratios().mean(axis=1),
    "MaxRReq": lambda v: v.requirement_ratios().max(axis=1),
    "MinRReq": lambda v: v.requirement_ratios().min(axis=1),
    "PlannedStart": _planned_start,
    "Delay": _delay,
    "Remaining": lambda v: v.remaining_durations / max(int(v.instance.durations.max()), 1),
    "Interrupted": lambda v: v.state.was_interrupted[v.candidates].astype(float),
    "TimePressure": lambda v: (v.instance.latest_start[v.candidates] - v.candidate_start_days)
    / _lower_bound(v),
    "AvailabilityRatio": _availability_ratio,
    "TimeToRecovery": _time_to_recovery,
}
