"""Time-stepped execution of a project with a pluggable priority rule.

Parallel SGS (non-delay): at each time t, repeatedly start the eligible activity
with the best rule value until nothing else fits, then advance t.

Optional ``baseline`` start times act as release dates when ``no_early_start`` is
True (no activity may start earlier than planned). With the baseline order as the
rule and no disturbances, the simulation reproduces the baseline exactly.

Disruptions (availability changes, violation check, Chand repair) are the next
step; the ``avail`` array already exists so they can plug in without changes to
the dispatch logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .instance import Instance
from .rules import Rule

INF = 10 ** 9


@dataclass
class SimState:
    t: int
    start: np.ndarray        # -1 = not started
    finish: np.ndarray       # INF = not started
    usage: np.ndarray        # [H, K] committed resource usage of started activities
    avail: np.ndarray        # [H, K] currently known availability


@dataclass
class Result:
    start: np.ndarray
    finish: np.ndarray
    makespan: int


def simulate(inst: Instance, rule: Rule, baseline: Optional[np.ndarray] = None,
             no_early_start: bool = True, horizon: Optional[int] = None) -> Result:
    J, K = inst.n_jobs, inst.n_res
    H = (int(inst.d.sum()) + 1) if horizon is None else horizon
    state = SimState(
        t=0,
        start=np.full(J, -1, dtype=np.int64),
        finish=np.full(J, INF, dtype=np.int64),
        usage=np.zeros((H, K), dtype=np.int64),
        avail=np.tile(inst.cap, (H, 1)),
    )
    release = baseline if (baseline is not None and no_early_start) else None
    n_started = 0

    def eligible(t: int) -> np.ndarray:
        out = []
        for j in range(J):
            if state.start[j] >= 0:
                continue
            if release is not None and t < release[j]:
                continue
            if any(state.finish[p] > t for p in inst.pred[j]):
                continue
            dj = int(inst.d[j])
            if dj > 0 and not np.all(state.usage[t:t + dj] + inst.req[j] <= state.avail[t:t + dj]):
                continue
            out.append(j)
        return np.array(out, dtype=np.int64)

    t = 0
    while n_started < J:
        if t >= H - 1:
            raise RuntimeError("simulation horizon exceeded")
        state.t = t
        while True:
            elig = eligible(t)
            if len(elig) == 0:
                break
            if len(elig) == 1:
                pick = int(elig[0])
            else:
                pr = np.asarray(rule(inst, state, elig), dtype=float)
                pick = int(elig[np.lexsort((elig, pr))[0]])   # lowest value, then lowest id
            dj = int(inst.d[pick])
            state.start[pick] = t
            state.finish[pick] = t + dj
            if dj > 0:
                state.usage[t:t + dj] += inst.req[pick]
            n_started += 1
        t += 1

    return Result(state.start.copy(), state.finish.copy(), int(state.finish[inst.sink]))
