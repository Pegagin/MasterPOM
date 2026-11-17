"""Stochastic resource disruptions after Chand et al. (2019), Algorithm 1.

Three setting variables (paper names in brackets):

* ``failure_probability`` (fp)  -- chance that a new disruption is *discovered* on a given day.
* ``mean_repair_time``    (ATR) -- mean of the exponential disruption duration (at least 1 day).
* ``mean_lead_time``      (ATF) -- mean of the exponential delay between discovery and start.
  ``0`` (the default for now) means every disruption starts on the day it is discovered.

A disruption hits a uniformly random number (1..K) of distinct resources. Each affected
resource loses a uniformly random fraction of its *current* availability for the duration,
so overlapping disruptions compound. Availability is rounded down to whole units.

Every day has its own random stream derived from ``(seed, day)``. What happens on a day
therefore never depends on the schedule, on which days were asked before, or on which
heuristic is being simulated: all heuristics face exactly the same disruptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Values used for the dynamic experiments in Chand et al. (2019), Section 5.1.2
CHAND_FAILURE_PROBABILITIES = (0.15, 0.20, 0.25)
CHAND_MEAN_REPAIR_TIMES = (1, 2, 3)
CHAND_MEAN_LEAD_TIMES = (4, 6, 8)


@dataclass(frozen=True)
class DisruptionSettings:
    failure_probability: float = 0.2
    mean_repair_time: float = 2.0
    mean_lead_time: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.failure_probability <= 1.0:
            raise ValueError("failure_probability must be between 0 and 1")
        if self.mean_repair_time <= 0 or self.mean_lead_time < 0:
            raise ValueError("mean_repair_time must be > 0 and mean_lead_time >= 0")


@dataclass(frozen=True)
class Disruption:
    discovery_day: int
    start_day: int
    duration: int
    resources: Tuple[int, ...]                 # affected resource ids
    reduction_fractions: Tuple[float, ...]     # share of current availability lost, per resource

    @property
    def end_day(self) -> int:
        """First day after the disruption."""
        return self.start_day + self.duration


class DisruptionGenerator:
    """Reproducible day-by-day disruption stream for one project run."""

    def __init__(self, n_resources: int, settings: DisruptionSettings, seed: int):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.n_resources = n_resources
        self.settings = settings
        self.seed = seed

    def disruptions_discovered_on(self, day: int) -> List[Disruption]:
        """The disruptions discovered on ``day`` (zero or one, as in Chand's Algorithm 1)."""
        rng = np.random.default_rng([self.seed, day])
        if rng.random() >= self.settings.failure_probability:
            return []
        lead_time = 0
        if self.settings.mean_lead_time > 0:
            lead_time = int(np.floor(rng.exponential(self.settings.mean_lead_time)))
        duration = max(1, int(round(rng.exponential(self.settings.mean_repair_time))))
        n_affected = int(rng.integers(1, self.n_resources + 1))
        resources = tuple(int(k) for k in sorted(rng.choice(self.n_resources, n_affected,
                                                            replace=False)))
        reduction_fractions = tuple(float(f) for f in rng.random(n_affected))
        return [Disruption(day, day + lead_time, duration, resources, reduction_fractions)]
