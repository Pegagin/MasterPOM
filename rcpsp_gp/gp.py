"""Genetic-programming representation of priority heuristics (DEAP).

An evolved heuristic is an arithmetic expression tree over the terminals of
:mod:`rcpsp_gp.decision` with Chand et al.'s operator set (Table 2):

    add, sub, mul, div (protected: x / 0 = 0), max, min, neg

for example ``sub(LF, AvgRReq)``. Applied to a decision it returns one priority per candidate
(lower = scheduled first), so it can be used anywhere a standard rule can.

Heuristics are saved as JSON: the expression strings plus the terminal list and the
settings they were trained with. The expression only makes sense with the same terminal
definitions, so the file also stores ``TERMINAL_SET_VERSION`` and loading checks it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from deap import base, creator, gp

from .decision import CHAND_TERMINALS, EXTRA_TERMINALS, DecisionView

TERMINAL_SET_VERSION = "decision-terminals-v1"


def terminal_names(use_extra_terminals: bool = False) -> Tuple[str, ...]:
    return CHAND_TERMINALS + (EXTRA_TERMINALS if use_extra_terminals else ())


def protected_division(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``numerator / denominator``, but 0 wherever the denominator is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
    return np.where(np.asarray(denominator) == 0, 0.0, result)


def build_primitive_set(terminals: Sequence[str]) -> gp.PrimitiveSet:
    primitive_set = gp.PrimitiveSet("PRIORITY", len(terminals))
    primitive_set.renameArguments(**{f"ARG{i}": name for i, name in enumerate(terminals)})
    primitive_set.addPrimitive(np.add, 2, name="add")
    primitive_set.addPrimitive(np.subtract, 2, name="sub")
    primitive_set.addPrimitive(np.multiply, 2, name="mul")
    primitive_set.addPrimitive(protected_division, 2, name="div")
    primitive_set.addPrimitive(np.maximum, 2, name="max")
    primitive_set.addPrimitive(np.minimum, 2, name="min")
    primitive_set.addPrimitive(np.negative, 1, name="neg")
    return primitive_set


def ensure_deap_types() -> None:
    """Create DEAP's fitness and individual classes once (minimisation)."""
    if not hasattr(creator, "PriorityFitness"):
        creator.create("PriorityFitness", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "PriorityTree"):
        creator.create("PriorityTree", gp.PrimitiveTree, fitness=creator.PriorityFitness)


class EvolvedHeuristic:
    """A compiled expression tree usable as a priority rule."""

    def __init__(self, tree: gp.PrimitiveTree, primitive_set: gp.PrimitiveSet):
        self.tree = tree
        self.expression = str(tree)
        self.terminals = tuple(primitive_set.arguments)
        # DEAP keeps a renamed argument's new name in ``value`` (``name`` stays "ARGi")
        self.used_terminals = {str(node.value) for node in tree if isinstance(node, gp.Terminal)}
        self._function = gp.compile(tree, primitive_set)

    @classmethod
    def from_expression(cls, expression: str, primitive_set: gp.PrimitiveSet) -> "EvolvedHeuristic":
        return cls(gp.PrimitiveTree.from_string(expression, primitive_set), primitive_set)

    @property
    def name(self) -> str:
        return f"GP[{self.expression}]"

    @property
    def size(self) -> int:
        """Number of nodes (Chand's complexity measure)."""
        return len(self.tree)

    def __call__(self, view: DecisionView) -> np.ndarray:
        arguments = [view.terminal(name) if name in self.used_terminals else None
                     for name in self.terminals]
        with np.errstate(all="ignore"):
            priorities = self._function(*arguments)
        return np.broadcast_to(np.asarray(priorities, dtype=float), view.candidates.shape)

    def __repr__(self) -> str:
        return self.name


# ------------------------------------------------------------------------------ files
def save_heuristics(path, heuristics: Sequence[EvolvedHeuristic],
                    validation_scores: Optional[Sequence[float]] = None,
                    metadata: Optional[Dict] = None) -> Path:
    """Write heuristics, best first, to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if validation_scores is None:
        validation_scores = [None] * len(heuristics)
    document = {
        "terminal_set_version": TERMINAL_SET_VERSION,
        "terminals": list(heuristics[0].terminals) if heuristics else [],
        "metadata": metadata or {},
        "heuristics": [{"rank": rank, "expression": heuristic.expression,
                        "size": heuristic.size, "validation_score": score}
                       for rank, (heuristic, score)
                       in enumerate(zip(heuristics, validation_scores), start=1)],
    }
    path.write_text(json.dumps(document, indent=1))
    return path


def load_heuristics(path) -> Tuple[List[EvolvedHeuristic], Dict]:
    """Read a file written by :func:`save_heuristics`; returns (heuristics, whole document)."""
    document = json.loads(Path(path).read_text())
    if document.get("terminal_set_version") != TERMINAL_SET_VERSION:
        raise ValueError(f"{path} was written with terminal set "
                         f"{document.get('terminal_set_version')!r}, this code uses "
                         f"{TERMINAL_SET_VERSION!r}")
    primitive_set = build_primitive_set(document["terminals"])
    heuristics = [EvolvedHeuristic.from_expression(entry["expression"], primitive_set)
                  for entry in document["heuristics"]]
    return heuristics, document
