"""Baseline (predictive) schedule generation via the serial SGS."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple

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


def default_time_limit(inst: Instance, per_job: float = 0.5, minimum: float = 5.0) -> float:
    """Time budget that grows with problem size (about J30: 16 s, J60: 31 s, J120: 61 s)."""
    return max(minimum, per_job * inst.n_jobs)


def cpsat_solve(inst: Instance, time_limit: Optional[float] = None, workers: int = 8,
                seed: int = 0, warm_start: bool = True, gap: float = 0.0
                ) -> Tuple[np.ndarray, str, int]:
    """Like :func:`cpsat_baseline` but returns ``(start, status, bound)``.

    ``status`` is ``OPTIMAL`` (proven), ``FEASIBLE`` (stopped by time/gap limit) or
    ``SAMPLING`` (CP-SAT found nothing better than the warm start). ``bound`` is the best
    proven lower bound on the makespan (at least the critical-path ``inst.lb``).
    """
    return _cpsat(inst, time_limit, workers, seed, warm_start, gap)


def cpsat_baseline(inst: Instance, time_limit: Optional[float] = None, workers: int = 8,
                   seed: int = 0, warm_start: bool = True, gap: float = 0.0) -> np.ndarray:
    """Minimum-makespan schedule from OR-tools CP-SAT (interval + cumulative model).

    Stops as soon as optimality is proven (or the relative ``gap`` to the bound is reached),
    otherwise at ``time_limit`` seconds; ``None`` scales the limit with instance size, so small
    problems end up solved to optimality and large ones are time-limited.
    Optionally warm-started with the random-sampling schedule, so the result is never worse.
    """
    return _cpsat(inst, time_limit, workers, seed, warm_start, gap)[0]


def _cpsat(inst, time_limit, workers, seed, warm_start, gap):
    if time_limit is None:
        time_limit = default_time_limit(inst)
    from ortools.sat.python import cp_model  # lazy: only needed for this method

    J, K = inst.n_jobs, inst.n_res
    horizon = int(inst.d.sum())
    hint = random_sampling_baseline(inst) if warm_start else None
    if hint is not None:
        horizon = min(horizon, makespan(inst, hint))   # hint is feasible, so this is valid

    m = cp_model.CpModel()
    starts = [m.NewIntVar(0, horizon, f"s{j}") for j in range(J)]
    intervals = [m.NewIntervalVar(starts[j], int(inst.d[j]), starts[j] + int(inst.d[j]), f"i{j}")
                 for j in range(J)]
    for j in range(J):
        for p in inst.pred[j]:
            m.Add(starts[j] >= starts[p] + int(inst.d[p]))
    for k in range(K):
        use = [j for j in range(J) if inst.d[j] > 0 and inst.req[j, k] > 0]
        m.AddCumulative([intervals[j] for j in use],
                        [int(inst.req[j, k]) for j in use], int(inst.cap[k]))
    m.Minimize(starts[inst.sink])
    if hint is not None:
        for j in range(J):
            m.AddHint(starts[j], int(hint[j]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.relative_gap_limit = float(gap)
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if hint is not None:
            return hint, "SAMPLING", inst.lb
        raise RuntimeError(f"CP-SAT found no schedule ({solver.StatusName(status)})")
    start = np.array([solver.Value(v) for v in starts], dtype=np.int64)
    bound = max(inst.lb, int(math.ceil(solver.BestObjectiveBound() - 1e-6)))
    return start, solver.StatusName(status), bound


def save_baseline(path, inst: Instance, start: np.ndarray, method: str,
                  status: str = "-", bound: Optional[int] = None) -> Path:
    """Write a baseline schedule to ``path`` as JSON (start times plus metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "instance": inst.name,
        "method": method,
        "status": status,
        "makespan": int(makespan(inst, start)),
        "lb": int(inst.lb),
        "bound": int(inst.lb if bound is None else bound),
        "start": [int(x) for x in start],
    }, indent=1))
    return path


def load_baseline(path) -> dict:
    """Read a file written by :func:`save_baseline`; ``["start"]`` is a numpy int array."""
    data = json.loads(Path(path).read_text())
    data["start"] = np.array(data["start"], dtype=np.int64)
    return data
