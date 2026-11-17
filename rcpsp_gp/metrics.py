"""Schedule validation and performance metrics."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .instance import Instance


def makespan(instance: Instance, start_days: np.ndarray) -> int:
    return int(start_days[instance.sink] + instance.durations[instance.sink])


def percent_over_lower_bound(instance: Instance, project_makespan: int) -> float:
    """Percentage deviation of a makespan from the critical-path lower bound (Chand's measure)."""
    return 100.0 * (project_makespan - instance.lower_bound) / max(instance.lower_bound, 1)


def pct_dev_lb(instance: Instance, start_days: np.ndarray) -> float:
    """Percentage deviation of a schedule's makespan from the critical-path lower bound."""
    return percent_over_lower_bound(instance, makespan(instance, start_days))


def stability(start_days: np.ndarray, baseline_start_days: np.ndarray,
              weights: Optional[np.ndarray] = None) -> float:
    """Weighted sum of absolute start-day deviations from the baseline."""
    deviation = np.abs(np.asarray(start_days) - np.asarray(baseline_start_days))
    if weights is not None:
        deviation = deviation * weights
    return float(deviation.sum())


def check_feasible(instance: Instance, start_days: np.ndarray,
                   capacities: Optional[np.ndarray] = None) -> List[str]:
    """Return a list of violations (empty list = feasible). Uses nominal capacity by default.

    ``capacities`` may also be a [days, n_resources] availability profile.
    """
    start_days = np.asarray(start_days)
    errors: List[str] = []
    if (start_days < 0).any():
        errors.append("negative or unset start day")
        return errors
    for activity in range(instance.n_activities):
        for successor in instance.successors[activity]:
            if start_days[successor] < start_days[activity] + instance.durations[activity]:
                errors.append(f"precedence {activity}->{successor} violated")
    horizon = int((start_days + instance.durations).max()) + 1
    usage = np.zeros((horizon, instance.n_resources), dtype=np.int64)
    for activity in range(instance.n_activities):
        if instance.durations[activity] > 0:
            usage[start_days[activity]:start_days[activity] + instance.durations[activity]] += \
                instance.requirements[activity]
    limit = instance.capacities if capacities is None else np.asarray(capacities)
    if limit.ndim == 2:          # availability profile; days beyond it have nominal capacity
        profile = np.tile(instance.capacities, (horizon, 1))
        days = min(horizon, len(limit))
        profile[:days] = limit[:days]
        limit = profile
    over = np.argwhere(usage > limit)
    for day, k in over[:5]:
        allowed = limit[day, k] if limit.ndim == 2 else limit[k]
        errors.append(f"resource {k} over capacity on day {day}: {usage[day, k]} > {allowed}")
    return errors
