"""Fixed train/test split of PSPLIB instances.

File names look like ``j3021_4.sm``: parameter set 21, instance 4 of that set. Instances
1..5 of every parameter set are for training and 6..10 for testing, so both halves cover
all difficulty regimes and the split needs no random seed or saved file.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

N_TRAIN = 5


def parse_name(stem: str) -> Tuple[int, int]:
    """``'j3021_4'`` -> ``(21, 4)`` (parameter set, instance number)."""
    m = re.match(r"^j(?:30|60|90|120)(\d+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"not a PSPLIB instance name: {stem!r}")
    return int(m.group(1)), int(m.group(2))


def split_instances(folder) -> Tuple[List[Path], List[Path]]:
    """Return ``(train, test)`` lists of .sm files in ``folder``, each sorted."""
    train, test = [], []
    for f in sorted(Path(folder).glob("*.sm")):
        _, k = parse_name(f.stem)
        (train if k <= N_TRAIN else test).append(f)
    key = lambda f: parse_name(f.stem)
    return sorted(train, key=key), sorted(test, key=key)
