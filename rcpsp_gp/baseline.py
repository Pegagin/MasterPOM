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
    """Static serial schedule generation scheme with nominal capacities (for baselines).

    ``priority`` is a per-activity value, lower = scheduled earlier (ties: lower id).
    Default is LFT. Returns the start-day vector.
    """
    n_activities = inst.n_activities
    if priority is None:
        priority = inst.latest_finish.astype(float)
    horizon = int(inst.durations.sum()) + 1
    usage = np.zeros((horizon, inst.n_resources), dtype=np.int64)
    start = np.full(n_activities, -1, dtype=np.int64)
    scheduled = np.zeros(n_activities, dtype=bool)

    for _ in range(n_activities):
        eligible = [j for j in range(n_activities)
                    if not scheduled[j] and all(scheduled[p] for p in inst.predecessors[j])]
        activity = min(eligible, key=lambda j: (priority[j], j))
        day = max((start[p] + inst.durations[p] for p in inst.predecessors[activity]), default=0)
        duration = int(inst.durations[activity])
        if duration > 0:
            requirement = inst.requirements[activity]
            while not np.all(usage[day:day + duration] + requirement <= inst.capacities):
                day += 1
            usage[day:day + duration] += requirement
        start[activity] = day
        scheduled[activity] = True
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
    spread = noise * max(1, int(inst.latest_finish.max() - inst.latest_finish.min()))
    for _ in range(n_samples - 1):
        pr = inst.latest_finish + rng.uniform(0, spread, size=inst.n_activities)
        s = serial_sgs(inst, pr)
        ms = makespan(inst, s)
        if ms < best_ms:
            best, best_ms = s, ms
            if best_ms == inst.lower_bound:
                break
    return best


def default_time_limit(inst: Instance, per_job: float = 0.5, minimum: float = 5.0) -> float:
    """Time budget that grows with problem size (about J30: 16 s, J60: 31 s, J120: 61 s)."""
    return max(minimum, per_job * inst.n_activities)


def cpsat_solve(inst: Instance, time_limit: Optional[float] = None, workers: int = 8,
                seed: int = 0, warm_start: bool = True, gap: float = 0.0
                ) -> Tuple[np.ndarray, str, int]:
    """Like :func:`cpsat_baseline` but returns ``(start, status, bound)``.

    ``status`` is ``OPTIMAL`` (proven), ``FEASIBLE`` (stopped by time/gap limit) or
    ``SAMPLING`` (CP-SAT found nothing better than the warm start). ``bound`` is the best
    proven lower bound on the makespan (at least the critical-path ``inst.lower_bound``).
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

    J, K = inst.n_activities, inst.n_resources
    d, req, cap = inst.durations, inst.requirements, inst.capacities
    horizon = int(d.sum())
    hint = random_sampling_baseline(inst) if warm_start else None
    if hint is not None:
        horizon = min(horizon, makespan(inst, hint))   # hint is feasible, so this is valid

    m = cp_model.CpModel()
    starts = [m.NewIntVar(0, horizon, f"s{j}") for j in range(J)]
    intervals = [m.NewIntervalVar(starts[j], int(d[j]), starts[j] + int(d[j]), f"i{j}")
                 for j in range(J)]
    for j in range(J):
        for p in inst.predecessors[j]:
            m.Add(starts[j] >= starts[p] + int(d[p]))
    for k in range(K):
        use = [j for j in range(J) if d[j] > 0 and req[j, k] > 0]
        m.AddCumulative([intervals[j] for j in use],
                        [int(req[j, k]) for j in use], int(cap[k]))
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
            return hint, "SAMPLING", inst.lower_bound
        raise RuntimeError(f"CP-SAT found no schedule ({solver.StatusName(status)})")
    start = np.array([solver.Value(v) for v in starts], dtype=np.int64)
    bound = max(inst.lower_bound, int(math.ceil(solver.BestObjectiveBound() - 1e-6)))
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
        "lb": int(inst.lower_bound),
        "bound": int(inst.lower_bound if bound is None else bound),
        "start": [int(x) for x in start],
    }, indent=1))
    return path


def load_baseline(path) -> dict:
    """Read a file written by :func:`save_baseline`; ``["start"]`` is a numpy int array."""
    data = json.loads(Path(path).read_text())
    data["start"] = np.array(data["start"], dtype=np.int64)
    return data
