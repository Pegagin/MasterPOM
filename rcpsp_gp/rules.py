"""Priority rules. A rule maps (instance, state, eligible ids) -> priority values.

Lower value = start first. Ties are broken by lower activity id in the simulator.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .instance import Instance

# (instance, SimState, eligible activity ids) -> priority per eligible activity
Rule = Callable[[Instance, "object", Sequence[int]], np.ndarray]


def _static(attr: str, sign: float = 1.0) -> Rule:
    def rule(inst: Instance, state, elig: Sequence[int]) -> np.ndarray:
        return sign * getattr(inst, attr)[list(elig)].astype(float)
    return rule


EST = _static("es")
EFT = _static("ef")
LST = _static("ls")
LFT = _static("lf")


def SPT(inst: Instance, state, elig: Sequence[int]) -> np.ndarray:
    return inst.d[list(elig)].astype(float)


def baseline_order(baseline_start: np.ndarray) -> Rule:
    """Priority = planned baseline start (i.e. H = baseline order)."""
    bs = np.asarray(baseline_start, dtype=float)

    def rule(inst: Instance, state, elig: Sequence[int]) -> np.ndarray:
        return bs[list(elig)]
    return rule


BENCHMARK_RULES = {"EST": EST, "EFT": EFT, "LST": LST, "LFT": LFT, "SPT": SPT}
