"""Fixed train / validation / test split of PSPLIB instances.

File names look like ``j3021_4.sm``: size 30, parameter set 21, instance 4 of that set.
Every parameter set has 10 instances, so splitting by instance number keeps all difficulty
regimes in every subset and needs no random seed or saved file.

Schemes
-------
``chand`` (default, Chand et al. 2019)
    instances 1-2 of every set -> train, 3 -> validation, 4-10 -> test. J90 and J120 are
    used for testing only (the paper trains on J30/J60 and tests generalisation to larger
    projects).
``half``
    instances 1-5 -> train, 6-10 -> test, no validation subset.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple

SPLIT_SCHEMES = ("chand", "half")
SUBSETS = ("train", "validation", "test")
TEST_ONLY_SIZES_IN_CHAND_SCHEME = (90, 120)


class InstanceName(NamedTuple):
    size: int               # number of real activities (30, 60, 90, 120)
    parameter_set: int      # PSPLIB parameter combination (NC, RF, RS)
    instance_number: int    # 1..10 within the parameter set


def parse_instance_name(stem: str) -> InstanceName:
    """``'j3021_4'`` -> ``InstanceName(size=30, parameter_set=21, instance_number=4)``."""
    match = re.match(r"^j(30|60|90|120)(\d+)_(\d+)$", stem)
    if not match:
        raise ValueError(f"not a PSPLIB instance name: {stem!r}")
    return InstanceName(int(match.group(1)), int(match.group(2)), int(match.group(3)))


def parse_name(stem: str):
    """``'j3021_4'`` -> ``(21, 4)`` (parameter set, instance number)."""
    name = parse_instance_name(stem)
    return name.parameter_set, name.instance_number


def subset_of(name: InstanceName, scheme: str = "chand") -> str:
    """Which subset ('train', 'validation' or 'test') an instance belongs to."""
    if scheme == "chand":
        if name.size in TEST_ONLY_SIZES_IN_CHAND_SCHEME or name.instance_number >= 4:
            return "test"
        return "train" if name.instance_number <= 2 else "validation"
    if scheme == "half":
        return "train" if name.instance_number <= 5 else "test"
    raise ValueError(f"unknown split scheme {scheme!r}; choose one of {SPLIT_SCHEMES}")


def split_instances(folder, scheme: str = "chand") -> Dict[str, List[Path]]:
    """Return ``{'train': [...], 'validation': [...], 'test': [...]}`` lists of .sm files."""
    subsets: Dict[str, List[Path]] = {subset: [] for subset in SUBSETS}
    for path in Path(folder).glob("*.sm"):
        subsets[subset_of(parse_instance_name(path.stem), scheme)].append(path)
    for files in subsets.values():
        files.sort(key=lambda path: parse_instance_name(path.stem))
    return subsets
