"""Schedule validation and performance metrics."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .instance import Instance


def makespan(inst: Instance, start: np.ndarray) -> int:
    return int(start[inst.sink] + inst.d[inst.sink])


def pct_dev_lb(inst: Instance, start: np.ndarray) -> float:
    """Percentage deviation of the makespan from the critical-path lower bound."""
    return 100.0 * (makespan(inst, start) - inst.lb) / inst.lb


def stability(start: np.ndarray, baseline_start: np.ndarray,
              weights: Optional[np.ndarray] = None) -> float:
    """Weighted sum of absolute start-time deviations from the baseline."""
    dev = np.abs(np.asarray(start) - np.asarray(baseline_start))
    if weights is not None:
        dev = dev * weights
    return float(dev.sum())


def check_feasible(inst: Instance, start: np.ndarray, cap: Optional[np.ndarray] = None) -> List[str]:
    """Return a list of violations (empty list = feasible). Uses nominal capacity by default."""
    cap = inst.cap if cap is None else cap
    errors: List[str] = []
    start = np.asarray(start)
    if (start < 0).any():
        errors.append("negative or unset start time")
        return errors
    for j in range(inst.n_jobs):
        for s in inst.succ[j]:
            if start[s] < start[j] + inst.d[j]:
                errors.append(f"precedence {j}->{s} violated")
    horizon = int((start + inst.d).max()) + 1
    usage = np.zeros((horizon, inst.n_res), dtype=np.int64)
    for j in range(inst.n_jobs):
        if inst.d[j] > 0:
            usage[start[j]:start[j] + inst.d[j]] += inst.req[j]
    over = np.argwhere(usage > cap)
    for t, k in over[:5]:
        errors.append(f"resource {k} over capacity at t={t}: {usage[t, k]} > {cap[k]}")
    return errors
