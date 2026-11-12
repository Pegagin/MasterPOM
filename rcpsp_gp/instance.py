"""RCPSP instance container, PSPLIB .sm parser and static attributes.

Conventions
-----------
* Activities are 0-indexed. Activity 0 is the dummy source, J-1 the dummy sink
  (PSPLIB job k is index k-1).
* Single-mode, renewable resources only.
* ``lb`` is the critical-path lower bound (ignores resources).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


@dataclass
class Instance:
    name: str
    d: np.ndarray            # [J] durations
    req: np.ndarray          # [J, K] resource requirements
    cap: np.ndarray          # [K] nominal capacities
    succ: List[List[int]]
    pred: List[List[int]]
    es: np.ndarray           # earliest start (critical path)
    ef: np.ndarray
    ls: np.ndarray           # latest start w.r.t. lb
    lf: np.ndarray
    lb: int                  # critical-path lower bound on makespan

    @property
    def n_jobs(self) -> int:
        return len(self.d)

    @property
    def n_res(self) -> int:
        return len(self.cap)

    @property
    def source(self) -> int:
        return 0

    @property
    def sink(self) -> int:
        return len(self.d) - 1


def topological_order(succ: Sequence[Sequence[int]], pred: Sequence[Sequence[int]]) -> List[int]:
    indeg = [len(p) for p in pred]
    stack = [j for j, k in enumerate(indeg) if k == 0]
    order: List[int] = []
    while stack:
        j = stack.pop()
        order.append(j)
        for s in succ[j]:
            indeg[s] -= 1
            if indeg[s] == 0:
                stack.append(s)
    if len(order) != len(succ):
        raise ValueError("precedence graph contains a cycle")
    return order


def build_instance(name: str, d, req, cap, succ: Sequence[Sequence[int]]) -> Instance:
    """Create an Instance and compute predecessors and static time attributes."""
    d = np.asarray(d, dtype=np.int64)
    req = np.asarray(req, dtype=np.int64)
    cap = np.asarray(cap, dtype=np.int64)
    J = len(d)
    succ_l = [sorted(set(int(s) for s in ss)) for ss in succ]
    pred_l: List[List[int]] = [[] for _ in range(J)]
    for j, ss in enumerate(succ_l):
        for s in ss:
            pred_l[s].append(j)
    order = topological_order(succ_l, pred_l)

    es = np.zeros(J, dtype=np.int64)
    for j in order:
        for p in pred_l[j]:
            es[j] = max(es[j], es[p] + d[p])
    ef = es + d
    lb = int(ef.max())

    lf = np.full(J, lb, dtype=np.int64)
    for j in reversed(order):
        for s in succ_l[j]:
            lf[j] = min(lf[j], lf[s] - d[s])
    ls = lf - d
    return Instance(name, d, req, cap, succ_l, pred_l, es, ef, ls, lf, lb)


# --------------------------------------------------------------------------
# PSPLIB .sm parser
# --------------------------------------------------------------------------
def _rows(lines: List[str], header_idx: int, count: int) -> List[List[int]]:
    """Collect `count` numeric rows after the line at header_idx."""
    out: List[List[int]] = []
    i = header_idx + 1
    while len(out) < count and i < len(lines):
        s = lines[i].strip()
        if s and (s[0].isdigit()):
            out.append([int(x) for x in s.split()])
        i += 1
    if len(out) != count:
        raise ValueError("unexpected end of file while reading a section")
    return out


def _find(lines: List[str], prefix: str) -> int:
    for i, ln in enumerate(lines):
        if ln.strip().startswith(prefix):
            return i
    raise ValueError(f"section '{prefix}' not found")


def parse_sm_text(text: str, name: str = "instance") -> Instance:
    lines = text.splitlines()
    m = re.search(r"jobs\s*\(incl\. supersource/sink\s*\)\s*:\s*(\d+)", text)
    if not m:
        raise ValueError("cannot find number of jobs")
    J = int(m.group(1))
    m = re.search(r"-\s*renewable\s*:\s*(\d+)", text)
    if not m:
        raise ValueError("cannot find number of renewable resources")
    K = int(m.group(1))

    prec = _rows(lines, _find(lines, "PRECEDENCE RELATIONS"), J)
    succ: List[List[int]] = [[] for _ in range(J)]
    for row in prec:
        job, modes, nsucc = row[0], row[1], row[2]
        if modes != 1:
            raise ValueError("only single-mode instances (.sm) are supported")
        succ[job - 1] = [s - 1 for s in row[3:3 + nsucc]]

    reqs = _rows(lines, _find(lines, "REQUESTS/DURATIONS"), J)
    d = np.zeros(J, dtype=np.int64)
    req = np.zeros((J, K), dtype=np.int64)
    for row in reqs:
        job, _mode, dur = row[0], row[1], row[2]
        d[job - 1] = dur
        req[job - 1] = row[3:3 + K]

    cap = np.array(_rows(lines, _find(lines, "RESOURCEAVAILABILITIES"), 1)[0][:K], dtype=np.int64)
    return build_instance(name, d, req, cap, succ)


def parse_sm(path) -> Instance:
    p = Path(path)
    return parse_sm_text(p.read_text(), name=p.stem)
