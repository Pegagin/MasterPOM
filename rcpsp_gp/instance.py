"""RCPSP instance container, PSPLIB .sm parser and static activity attributes.

Conventions
-----------
* Activities are 0-indexed. Activity 0 is the dummy source, the last activity the dummy
  sink (PSPLIB job k is index k-1).
* Single-mode, renewable resources only. Time is counted in whole days starting at 0.
* ``lower_bound`` is the critical-path lower bound on the makespan (it ignores resources).
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
    durations: np.ndarray              # [n_activities] days each activity takes
    requirements: np.ndarray           # [n_activities, n_resources] units needed per day
    capacities: np.ndarray             # [n_resources] nominal units available per day
    successors: List[List[int]]        # immediate successors of each activity
    predecessors: List[List[int]]      # immediate predecessors of each activity
    earliest_start: np.ndarray         # critical-path schedule, resources ignored
    earliest_finish: np.ndarray
    latest_start: np.ndarray           # latest start that still meets lower_bound
    latest_finish: np.ndarray
    lower_bound: int                   # critical-path length
    all_successors: np.ndarray         # [n, n] bool: all_successors[i, j] = j follows i (directly or not)

    @property
    def n_activities(self) -> int:
        return len(self.durations)

    @property
    def n_resources(self) -> int:
        return len(self.capacities)

    @property
    def source(self) -> int:
        return 0

    @property
    def sink(self) -> int:
        return len(self.durations) - 1

    @property
    def real_activities(self) -> np.ndarray:
        """Ids of all activities except the dummy source and sink."""
        return np.arange(1, self.n_activities - 1)

    @property
    def total_successor_count(self) -> np.ndarray:
        """Number of direct and indirect successors of each activity."""
        return self.all_successors.sum(axis=1)

    @property
    def total_predecessor_count(self) -> np.ndarray:
        """Number of direct and indirect predecessors of each activity."""
        return self.all_successors.sum(axis=0)


def topological_order(successors: Sequence[Sequence[int]],
                      predecessors: Sequence[Sequence[int]]) -> List[int]:
    open_predecessor_count = [len(p) for p in predecessors]
    ready = [j for j, count in enumerate(open_predecessor_count) if count == 0]
    order: List[int] = []
    while ready:
        activity = ready.pop()
        order.append(activity)
        for successor in successors[activity]:
            open_predecessor_count[successor] -= 1
            if open_predecessor_count[successor] == 0:
                ready.append(successor)
    if len(order) != len(successors):
        raise ValueError("precedence graph contains a cycle")
    return order


def build_instance(name: str, durations, requirements, capacities,
                   successors: Sequence[Sequence[int]]) -> Instance:
    """Create an Instance and compute predecessors and the static critical-path attributes."""
    durations = np.asarray(durations, dtype=np.int64)
    requirements = np.asarray(requirements, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    n_activities = len(durations)
    successor_lists = [sorted(set(int(s) for s in succ)) for succ in successors]
    predecessor_lists: List[List[int]] = [[] for _ in range(n_activities)]
    for activity, succ in enumerate(successor_lists):
        for successor in succ:
            predecessor_lists[successor].append(activity)
    order = topological_order(successor_lists, predecessor_lists)

    earliest_start = np.zeros(n_activities, dtype=np.int64)
    for activity in order:
        for predecessor in predecessor_lists[activity]:
            earliest_start[activity] = max(earliest_start[activity],
                                           earliest_start[predecessor] + durations[predecessor])
    earliest_finish = earliest_start + durations
    lower_bound = int(earliest_finish.max())

    latest_finish = np.full(n_activities, lower_bound, dtype=np.int64)
    for activity in reversed(order):
        for successor in successor_lists[activity]:
            latest_finish[activity] = min(latest_finish[activity],
                                          latest_finish[successor] - durations[successor])
    latest_start = latest_finish - durations

    all_successors = np.zeros((n_activities, n_activities), dtype=bool)
    for activity in reversed(order):
        for successor in successor_lists[activity]:
            all_successors[activity, successor] = True
            all_successors[activity] |= all_successors[successor]

    return Instance(name, durations, requirements, capacities, successor_lists, predecessor_lists,
                    earliest_start, earliest_finish, latest_start, latest_finish, lower_bound,
                    all_successors)


# --------------------------------------------------------------------------
# Instance complexity measures (Kolisch et al., used by PSPLIB)
# --------------------------------------------------------------------------
def resource_factor(instance: Instance) -> float:
    """Average share of the resource types a real activity needs (0..1)."""
    real = instance.real_activities
    return float((instance.requirements[real] > 0).mean())


def resource_strength(instance: Instance) -> float:
    """How generous the capacities are, averaged over resources (0 = tight, 1 = no conflicts).

    Per resource: (capacity - largest single requirement) / (peak demand of the earliest-start
    schedule - largest single requirement). A resource whose peak demand never exceeds the
    largest single requirement counts as 1.
    """
    n_days = int(instance.earliest_finish.max()) + 1
    demand = np.zeros((n_days, instance.n_resources), dtype=np.int64)
    for activity in range(instance.n_activities):
        demand[instance.earliest_start[activity]:instance.earliest_finish[activity]] += \
            instance.requirements[activity]
    largest_single_requirement = instance.requirements.max(axis=0)
    peak_demand = demand.max(axis=0)
    strengths = []
    for k in range(instance.n_resources):
        spread = peak_demand[k] - largest_single_requirement[k]
        strengths.append(1.0 if spread <= 0
                         else (instance.capacities[k] - largest_single_requirement[k]) / spread)
    return float(np.mean(strengths))


# --------------------------------------------------------------------------
# PSPLIB .sm parser
# --------------------------------------------------------------------------
def _numeric_rows(lines: List[str], header_index: int, count: int) -> List[List[int]]:
    """Collect ``count`` numeric rows after the line at ``header_index``."""
    rows: List[List[int]] = []
    i = header_index + 1
    while len(rows) < count and i < len(lines):
        text = lines[i].strip()
        if text and text[0].isdigit():
            rows.append([int(x) for x in text.split()])
        i += 1
    if len(rows) != count:
        raise ValueError("unexpected end of file while reading a section")
    return rows


def _find_section(lines: List[str], prefix: str) -> int:
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            return i
    raise ValueError(f"section '{prefix}' not found")


def parse_sm_text(text: str, name: str = "instance") -> Instance:
    lines = text.splitlines()
    match = re.search(r"jobs\s*\(incl\. supersource/sink\s*\)\s*:\s*(\d+)", text)
    if not match:
        raise ValueError("cannot find number of jobs")
    n_activities = int(match.group(1))
    match = re.search(r"-\s*renewable\s*:\s*(\d+)", text)
    if not match:
        raise ValueError("cannot find number of renewable resources")
    n_resources = int(match.group(1))

    precedence_rows = _numeric_rows(lines, _find_section(lines, "PRECEDENCE RELATIONS"), n_activities)
    successors: List[List[int]] = [[] for _ in range(n_activities)]
    for row in precedence_rows:
        job, n_modes, n_successors = row[0], row[1], row[2]
        if n_modes != 1:
            raise ValueError("only single-mode instances (.sm) are supported")
        successors[job - 1] = [s - 1 for s in row[3:3 + n_successors]]

    request_rows = _numeric_rows(lines, _find_section(lines, "REQUESTS/DURATIONS"), n_activities)
    durations = np.zeros(n_activities, dtype=np.int64)
    requirements = np.zeros((n_activities, n_resources), dtype=np.int64)
    for row in request_rows:
        job, duration = row[0], row[2]          # row[1] is the mode number (always 1)
        durations[job - 1] = duration
        requirements[job - 1] = row[3:3 + n_resources]

    capacity_row = _numeric_rows(lines, _find_section(lines, "RESOURCEAVAILABILITIES"), 1)[0]
    capacities = np.array(capacity_row[:n_resources], dtype=np.int64)
    return build_instance(name, durations, requirements, capacities, successors)


def parse_sm(path) -> Instance:
    path = Path(path)
    return parse_sm_text(path.read_text(), name=path.stem)
