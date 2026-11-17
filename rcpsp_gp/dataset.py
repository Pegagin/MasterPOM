"""Loading PSPLIB projects with their baselines, and building disruption scenarios.

A :class:`Project` bundles an instance with its near-optimal baseline plan (CP-SAT, cached
as JSON so it is computed only once) and its PSPLIB design parameters.

A :class:`Scenario` is one project run: a project plus the disruption settings and the
seed of the disruption stream. Every method evaluated on the same scenario meets the same
disruptions.

Resource-constrained filter (default on, Chand et al. 2019, Section 5.1.2): keep only
instances with resource strength RS <= 0.5 and resource factor RF > 0.5. RS and RF are
computed from the data and snapped to PSPLIB's design levels, because the computed RS of a
nominal 0.5 instance can come out at 0.52.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .baseline import cpsat_solve, load_baseline, save_baseline
from .disruption import (CHAND_FAILURE_PROBABILITIES, CHAND_MEAN_REPAIR_TIMES,
                         DisruptionGenerator, DisruptionSettings)
from .instance import Instance, parse_sm, resource_factor, resource_strength
from .split import parse_instance_name, split_instances

RESOURCE_FACTOR_LEVELS = (0.25, 0.5, 0.75, 1.0)
RESOURCE_STRENGTH_LEVELS = {30: (0.2, 0.5, 0.7, 1.0), 60: (0.2, 0.5, 0.7, 1.0),
                            90: (0.2, 0.5, 0.7, 1.0), 120: (0.1, 0.2, 0.3, 0.4, 0.5)}
MAX_RESOURCE_STRENGTH = 0.5          # filter: RS <= 0.5
MIN_RESOURCE_FACTOR = 0.5            # filter: RF > 0.5


def snap_to_level(value: float, levels: Sequence[float]) -> float:
    return float(min(levels, key=lambda level: abs(level - value)))


@dataclass
class Project:
    instance: Instance
    baseline_start: np.ndarray
    parameter_set: int
    instance_number: int
    resource_factor: float          # PSPLIB design level (snapped)
    resource_strength: float        # PSPLIB design level (snapped)
    baseline_status: str = "-"      # OPTIMAL / FEASIBLE / ... from CP-SAT

    @property
    def name(self) -> str:
        return self.instance.name

    @property
    def baseline_makespan(self) -> int:
        return int(self.baseline_start[self.instance.sink])

    @property
    def is_resource_constrained(self) -> bool:
        return (self.resource_strength <= MAX_RESOURCE_STRENGTH
                and self.resource_factor > MIN_RESOURCE_FACTOR)


def design_parameters(instance: Instance, size: int):
    """(resource factor, resource strength) snapped to PSPLIB's design levels."""
    strength_levels = RESOURCE_STRENGTH_LEVELS.get(size, RESOURCE_STRENGTH_LEVELS[30])
    return (snap_to_level(resource_factor(instance), RESOURCE_FACTOR_LEVELS),
            snap_to_level(resource_strength(instance), strength_levels))


def load_or_create_baseline(instance: Instance, cache_file: Path,
                            time_limit: Optional[float] = None):
    """Baseline start days from ``cache_file``; solved with CP-SAT and saved if missing."""
    if cache_file.exists():
        cached = load_baseline(cache_file)
        return cached["start"], cached.get("status", "-")
    start, status, bound = cpsat_solve(instance, time_limit=time_limit)
    save_baseline(cache_file, instance, start, "cpsat", status, bound)
    return start, status


def load_projects(data_folder, subset: str = "train", split_scheme: str = "chand",
                  resource_constrained_only: bool = True, baseline_folder="baselines",
                  baseline_time_limit: Optional[float] = None, limit: Optional[int] = None,
                  log=print) -> List[Project]:
    """Projects of one subset ('train', 'validation', 'test' or 'all').

    Baselines are read from ``<baseline_folder>/<data folder name>/<instance>.json`` and
    computed with CP-SAT the first time an instance is used.
    """
    data_folder = Path(data_folder)
    if subset == "all":
        files = sorted(data_folder.glob("*.sm"), key=lambda p: parse_instance_name(p.stem))
    else:
        files = split_instances(data_folder, split_scheme)[subset]
    cache_folder = Path(baseline_folder) / data_folder.name
    projects: List[Project] = []
    for path in files:
        name = parse_instance_name(path.stem)
        instance = parse_sm(path)
        factor, strength = design_parameters(instance, name.size)
        project_preview = Project(instance, np.zeros(0, dtype=np.int64), name.parameter_set,
                                  name.instance_number, factor, strength)
        if resource_constrained_only and not project_preview.is_resource_constrained:
            continue
        cache_file = cache_folder / f"{instance.name}.json"
        if not cache_file.exists() and log:
            log(f"  solving baseline for {instance.name} with CP-SAT ...")
        start, status = load_or_create_baseline(instance, cache_file, baseline_time_limit)
        project_preview.baseline_start = start
        project_preview.baseline_status = status
        projects.append(project_preview)
        if limit is not None and len(projects) >= limit:
            break
    return projects


# ------------------------------------------------------------------------------ scenarios
@dataclass(frozen=True)
class Scenario:
    project: Project
    disruption_settings: DisruptionSettings
    seed: int

    @property
    def label(self) -> str:
        settings = self.disruption_settings
        return (f"{self.project.name} seed={self.seed} fp={settings.failure_probability:g} "
                f"ATR={settings.mean_repair_time:g}")

    def disruption_generator(self) -> DisruptionGenerator:
        return DisruptionGenerator(self.project.instance.n_resources, self.disruption_settings,
                                   self.seed)


def make_scenarios(projects: Sequence[Project], replications: int = 3,
                   failure_probabilities: Sequence[float] = CHAND_FAILURE_PROBABILITIES,
                   mean_repair_times: Sequence[float] = CHAND_MEAN_REPAIR_TIMES,
                   mean_lead_time: float = 0.0, seed: int = 0) -> List[Scenario]:
    """``replications`` scenarios per project, each with its own seed and settings drawn
    uniformly from the given values (Chand et al. 2019, Section 5.1.2)."""
    rng = random.Random(seed)
    scenarios = []
    for project in projects:
        for _ in range(replications):
            settings = DisruptionSettings(failure_probability=rng.choice(failure_probabilities),
                                          mean_repair_time=rng.choice(mean_repair_times),
                                          mean_lead_time=mean_lead_time)
            scenarios.append(Scenario(project, settings, rng.randrange(2 ** 31)))
    return scenarios
