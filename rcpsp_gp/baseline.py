"""Baseline (predictive) schedule generation via the serial SGS."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .instance import Instance
from .metrics import makespan


def serial_sgs(inst: Instance, priority: Optional[np.ndarray] = None) -> np.ndarray:
    """Serial schedule generation scheme.

    ``priority`` is a per-activity value, lower = scheduled earlier (ties: lower id).
    Default is LFT. Returns the start-time vector.
    """
    J, K = inst.n_jobs, inst.n_res
    if priority is None:
        priority = inst.lf.astype(float)
    horizon = int(inst.d.sum()) + 1
    usage = np.zeros((horizon, K), dtype=np.int64)
    start = np.full(J, -1, dtype=np.int64)
    scheduled = np.zeros(J, dtype=bool)

    for _ in range(J):
        elig = [j for j in range(J)
                if not scheduled[j] and all(scheduled[p] for p in inst.pred[j])]
        j = min(elig, key=lambda x: (priority[x], x))
        t = max((start[p] + inst.d[p] for p in inst.pred[j]), default=0)
        dj = int(inst.d[j])
        if dj > 0:
            while not np.all(usage[t:t + dj] + inst.req[j] <= inst.cap):
                t += 1
            usage[t:t + dj] += inst.req[j]
        start[j] = t
        scheduled[j] = True
    return start


def random_sampling_baseline(inst: Instance, n_samples: int = 200, seed: int = 0,
                             noise: float = 0.5) -> np.ndarray:
    """Best of ``n_samples`` serial SGS runs with LFT priorities plus random noise.

    Sample 0 is the plain LFT schedule, so the result is never worse than LFT.
    Good enough as a tight baseline for J30/J60; swap in CP-SAT later if desired.
    """
    rng = np.random.default_rng(seed)
    best = serial_sgs(inst)
    best_ms = makespan(inst, best)
    spread = noise * max(1, int(inst.lf.max() - inst.lf.min()))
    for _ in range(n_samples - 1):
        pr = inst.lf + rng.uniform(0, spread, size=inst.n_jobs)
        s = serial_sgs(inst, pr)
        ms = makespan(inst, s)
        if ms < best_ms:
            best, best_ms = s, ms
            if best_ms == inst.lb:
                break
    return best
