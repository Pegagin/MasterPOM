"""Standard priority rules (Chand et al. 2019, Table 1) and the baseline-order rule.

A rule maps a :class:`~rcpsp_gp.decision.DecisionView` to one priority value per candidate.
**Lower value = scheduled first**; ties go to the lower activity id. Rules that the
literature defines as "maximise X" therefore return ``-X``.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .capacity import CapacityProfile
from .decision import DecisionView

Rule = Callable[[DecisionView], np.ndarray]


def rule_name(rule) -> str:
    return getattr(rule, "name", None) or getattr(rule, "__name__", repr(rule))


def _named(name: str):
    def decorate(function):
        function.name = name
        return function
    return decorate


@_named("EST")
def earliest_start_time(view: DecisionView) -> np.ndarray:
    return view.instance.earliest_start[view.candidates].astype(float)


@_named("EFT")
def earliest_finish_time(view: DecisionView) -> np.ndarray:
    return view.instance.earliest_finish[view.candidates].astype(float)


@_named("LST")
def latest_start_time(view: DecisionView) -> np.ndarray:
    return view.instance.latest_start[view.candidates].astype(float)


@_named("LFT")
def latest_finish_time(view: DecisionView) -> np.ndarray:
    return view.instance.latest_finish[view.candidates].astype(float)


@_named("SPT")
def shortest_processing_time(view: DecisionView) -> np.ndarray:
    """Shortest remaining duration first (equals the full duration unless resumed)."""
    return view.remaining_durations.astype(float)


@_named("MTS")
def most_total_successors(view: DecisionView) -> np.ndarray:
    return -view.instance.total_successor_count[view.candidates].astype(float)


@_named("FIFO")
def first_in_first_out(view: DecisionView) -> np.ndarray:
    """Lowest activity id first."""
    return view.candidates.astype(float)


@_named("GRPW")
def greatest_rank_positional_weight(view: DecisionView) -> np.ndarray:
    """Own duration plus the durations of all (direct and indirect) successors, largest first."""
    durations = view.instance.durations
    successor_work = view.instance.all_successors[view.candidates].astype(float) @ durations
    return -(durations[view.candidates] + successor_work)


@_named("GRD")
def greatest_resource_demand(view: DecisionView) -> np.ndarray:
    """Duration times total requirement over all resources, largest first."""
    total_requirement = view.instance.requirements[view.candidates].sum(axis=1)
    return -(view.instance.durations[view.candidates] * total_requirement).astype(float)


def earliest_start_given_other_started(view: DecisionView) -> np.ndarray:
    """``E[i, j]``: earliest day candidate ``j`` can start if candidate ``i`` starts now.

    "Now" is each candidate's ``candidate_start_days`` entry. Used by WCS and ACS
    (Kolisch 1996). The diagonal is unused and left at 0.
    """
    n = view.n_candidates
    earliest = np.zeros((n, n), dtype=np.int64)
    requirements = view.instance.requirements[view.candidates]
    for i in range(n):
        capacity_if_i_starts = view.free_capacity.copy()
        capacity_if_i_starts.reserve(requirements[i], int(view.candidate_start_days[i]),
                                     int(view.remaining_durations[i]))
        for j in range(n):
            if j != i:
                earliest[i, j] = _earliest_fit_or_end(capacity_if_i_starts, requirements[j],
                                                      int(view.candidate_start_days[j]),
                                                      int(view.remaining_durations[j]))
    return earliest


def _earliest_fit_or_end(capacity: CapacityProfile, requirement, from_day: int,
                         duration: int) -> int:
    """Like ``earliest_fit`` but tolerant of profiles that are over-committed for a while."""
    day = from_day
    limit = from_day + len(capacity.units_per_day) + duration + 1
    while day < limit and not capacity.fits(requirement, day, duration):
        day += 1
    return day


@_named("WCS")
def worst_case_slack(view: DecisionView) -> np.ndarray:
    """LS_j minus the latest day j could be pushed to by starting one other candidate now."""
    latest_start = view.instance.latest_start[view.candidates].astype(float)
    if view.n_candidates < 2:
        return latest_start
    earliest = earliest_start_given_other_started(view).astype(float)
    np.fill_diagonal(earliest, -np.inf)
    return latest_start - earliest.max(axis=0)


@_named("ACS")
def average_case_slack(view: DecisionView) -> np.ndarray:
    """LS_j minus the average day j could be pushed to by starting another candidate now."""
    latest_start = view.instance.latest_start[view.candidates].astype(float)
    n = view.n_candidates
    if n < 2:
        return latest_start
    earliest = earliest_start_given_other_started(view).astype(float)
    np.fill_diagonal(earliest, 0.0)
    return latest_start - earliest.sum(axis=0) / (n - 1)


@_named("BASELINE")
def baseline_order(view: DecisionView) -> np.ndarray:
    """Planned baseline start day first (the order of the predictive plan)."""
    baseline = view.state.baseline_start
    if baseline is None:
        raise ValueError("baseline_order needs a project with a baseline schedule")
    return baseline[view.candidates].astype(float)


STANDARD_RULES: Dict[str, Rule] = {
    rule.name: rule for rule in (
        earliest_start_time, earliest_finish_time, latest_start_time, latest_finish_time,
        shortest_processing_time, most_total_successors, first_in_first_out,
        greatest_rank_positional_weight, greatest_resource_demand,
        worst_case_slack, average_case_slack)
}

# Backwards-compatible alias used by older scripts
BENCHMARK_RULES = STANDARD_RULES
