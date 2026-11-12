"""Small random RCPSP instance generator (for tests and quick experiments).

Not a replacement for PSPLIB / ProGen; just enough structure to exercise code.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance, build_instance


def random_instance(n: int = 10, n_res: int = 2, seed: int = 0, edge_prob: float = 0.25,
                    max_dur: int = 10, tightness: float = 1.3) -> Instance:
    """n real activities + dummy source (0) and sink (n+1)."""
    rng = np.random.default_rng(seed)
    J = n + 2
    succ = [set() for _ in range(J)]
    for j in range(1, n + 1):
        for i in range(1, j):
            if rng.random() < edge_prob:
                succ[i].add(j)
    has_pred = [False] * J
    for i in range(J):
        for s in succ[i]:
            has_pred[s] = True
    for j in range(1, n + 1):
        if not has_pred[j]:
            succ[0].add(j)
        if not succ[j]:
            succ[j].add(J - 1)
    d = np.zeros(J, dtype=np.int64)
    d[1:n + 1] = rng.integers(1, max_dur + 1, size=n)
    req = np.zeros((J, n_res), dtype=np.int64)
    req[1:n + 1] = rng.integers(0, 6, size=(n, n_res))
    cap = np.maximum(req.max(axis=0), 1)
    cap = np.maximum(np.ceil(cap * rng.uniform(1.0, tightness, size=n_res)), 1).astype(np.int64)
    cap = np.maximum(cap, req.max(axis=0))
    return build_instance(f"rand_n{n}_s{seed}", d, req, cap, [sorted(s) for s in succ])
