"""Small random RCPSP instance generator (for tests and quick experiments).

Not a replacement for PSPLIB / ProGen; just enough structure to exercise code.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance, build_instance


def random_instance(n: int = 10, n_res: int = 2, seed: int = 0, edge_prob: float = 0.25,
                    max_dur: int = 10, tightness: float = 1.3) -> Instance:
    """``n`` real activities plus a dummy source (0) and sink (n+1), ``n_res`` resources."""
    rng = np.random.default_rng(seed)
    n_activities = n + 2
    successors = [set() for _ in range(n_activities)]
    for later in range(1, n + 1):
        for earlier in range(1, later):
            if rng.random() < edge_prob:
                successors[earlier].add(later)
    has_predecessor = [False] * n_activities
    for activity in range(n_activities):
        for successor in successors[activity]:
            has_predecessor[successor] = True
    for activity in range(1, n + 1):
        if not has_predecessor[activity]:
            successors[0].add(activity)
        if not successors[activity]:
            successors[activity].add(n_activities - 1)
    durations = np.zeros(n_activities, dtype=np.int64)
    durations[1:n + 1] = rng.integers(1, max_dur + 1, size=n)
    requirements = np.zeros((n_activities, n_res), dtype=np.int64)
    requirements[1:n + 1] = rng.integers(0, 6, size=(n, n_res))
    capacities = np.maximum(requirements.max(axis=0), 1)
    capacities = np.ceil(capacities * rng.uniform(1.0, tightness, size=n_res)).astype(np.int64)
    capacities = np.maximum(capacities, np.maximum(requirements.max(axis=0), 1))
    return build_instance(f"rand_n{n}_s{seed}", durations, requirements, capacities,
                          [sorted(s) for s in successors])
