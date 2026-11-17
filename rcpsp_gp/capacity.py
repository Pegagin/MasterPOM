"""Day-by-day resource profiles that grow on demand.

Planning and disruptions both work on ``[day, resource]`` tables whose length is not known in
advance. :class:`CapacityProfile` hides that: days beyond the stored table are assumed to
hold ``default_units`` (normally the nominal capacities, since nothing is known or planned
there yet).
"""
from __future__ import annotations

import numpy as np


def pad_days(table: np.ndarray, n_days: int, default_units: np.ndarray) -> np.ndarray:
    """Return ``table`` extended to at least ``n_days`` rows, new rows filled with ``default_units``."""
    if len(table) >= n_days:
        return table
    extra_rows = np.tile(np.asarray(default_units, dtype=table.dtype), (n_days - len(table), 1))
    return np.vstack([table, extra_rows])


class CapacityProfile:
    """Units of each resource per day (for example free capacity left for planning)."""

    def __init__(self, units_per_day: np.ndarray, default_units: np.ndarray):
        self.units_per_day = np.array(units_per_day, dtype=np.int64)
        self.default_units = np.asarray(default_units, dtype=np.int64)

    def copy(self) -> "CapacityProfile":
        return CapacityProfile(self.units_per_day, self.default_units)

    def window(self, start_day: int, n_days: int) -> np.ndarray:
        """Rows ``start_day .. start_day + n_days - 1`` (padded with default units)."""
        end_day = start_day + n_days
        if end_day > len(self.units_per_day):
            self.units_per_day = pad_days(self.units_per_day, end_day, self.default_units)
        return self.units_per_day[start_day:end_day]

    def fits(self, requirement: np.ndarray, start_day: int, duration: int) -> bool:
        """Whether ``requirement`` units are available on every day of the window."""
        if duration <= 0:
            return True
        return bool(np.all(self.window(start_day, duration) >= requirement))

    def reserve(self, requirement: np.ndarray, start_day: int, duration: int) -> None:
        """Take ``requirement`` units away on every day of the window."""
        if duration > 0:
            self.window(start_day, duration)            # make sure the rows exist
            self.units_per_day[start_day:start_day + duration] -= requirement

    def earliest_fit(self, requirement: np.ndarray, from_day: int, duration: int) -> int:
        """First day ``>= from_day`` on which ``requirement`` fits for ``duration`` days."""
        if np.any(requirement > self.default_units):
            raise ValueError("requirement exceeds the default units and can never fit")
        day = from_day
        while not self.fits(requirement, day, duration):
            day += 1
        return day
