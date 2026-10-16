"""Mini-Project 1: district population data model and classroom planning."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .forecasting import Forecaster, holdout_scores


class DistrictPopulation:
    """Annual population estimates for one district (values in thousands).

    Parameters
    ----------
    name : district name.
    years : calendar years, strictly increasing.
    population : population estimates in thousands, same length as ``years``.
    """

    def __init__(self, name: str, years: Sequence[int], population: Sequence[float]) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("District name must be a non-empty string")
        yrs = np.asarray(years, dtype=int)
        pop = np.asarray(population, dtype=float)
        if yrs.ndim != 1 or pop.ndim != 1:
            raise ValueError("years and population must be one-dimensional")
        if yrs.size != pop.size:
            raise ValueError(f"years ({yrs.size}) and population ({pop.size}) differ in length")
        if pop.size < 2:
            raise ValueError("Need at least two observations")
        if np.any(pop < 0):
            raise ValueError("Population cannot be negative")
        if np.any(np.diff(yrs) <= 0):
            raise ValueError("Years must be strictly increasing")
        self.name = name
        self.years = yrs
        self.population = pop

    def __repr__(self) -> str:
        return (f"DistrictPopulation(name={self.name!r}, years={self.years[0]}–{self.years[-1]}, "
                f"n={len(self)}, latest={self.population[-1]:,.0f}k)")

    def __len__(self) -> int:
        return int(self.population.size)

    # descriptive statistics ---------------------------------------------- #
    def stats_python(self) -> dict[str, float]:
        """Mean, median, *sample* variance and stdev via the ``statistics`` module."""
        data = self.population.tolist()
        return {
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "variance": statistics.variance(data),   # divides by n-1
            "stdev": statistics.stdev(data),
        }

    def stats_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Same statistics via NumPy. ``ddof=0`` (default) gives the *population* variance."""
        p = self.population
        return {
            "mean": float(np.mean(p)),
            "median": float(np.median(p)),
            "variance": float(np.var(p, ddof=ddof)),
            "stdev": float(np.std(p, ddof=ddof)),
        }

    # growth --------------------------------------------------------------- #
    def growth_rates(self) -> np.ndarray:
        """Year-on-year growth rates (fractions), length n-1."""
        p = self.population
        if np.any(p[:-1] == 0):
            raise ZeroDivisionError("Growth rate undefined after a zero population")
        return p[1:] / p[:-1] - 1

    def cagr(self) -> float:
        """Compound annual growth rate over the whole span."""
        p, y = self.population, self.years
        if p[0] <= 0:
            raise ValueError("CAGR undefined for a non-positive starting value")
        return float((p[-1] / p[0]) ** (1 / (y[-1] - y[0])) - 1)

    # train / test --------------------------------------------------------- #
    def split(self, last_train_year: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (train, test) population arrays split after ``last_train_year``."""
        mask = self.years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError("Split year leaves train or test empty")
        return self.population[mask], self.population[~mask]

    def evaluate(self, models: Sequence[Forecaster], last_train_year: int) -> dict[str, dict[str, float]]:
        """Hold-out MAE/RMSE/MAPE for each model."""
        train, test = self.split(last_train_year)
        return {m.name: holdout_scores(m, train, test) for m in models}


@dataclass(frozen=True)
class ClassroomPlanner:
    """Turns a population change into a classroom requirement.

    school_age_share : fraction of population of primary-school age (0–1).
    pupils_per_classroom : classroom capacity.
    unit : multiplier to convert population units to people (1000 for thousands).
    """

    school_age_share: float = 0.18
    pupils_per_classroom: int = 53
    unit: int = 1000

    def __post_init__(self) -> None:
        if not 0 < self.school_age_share <= 1:
            raise ValueError("school_age_share must be in (0, 1]")
        if self.pupils_per_classroom <= 0:
            raise ValueError("pupils_per_classroom must be positive")

    def additional_pupils(self, current: float, future: float) -> float:
        return (future - current) * self.unit * self.school_age_share

    def additional_classrooms(self, current: float, future: float) -> int:
        """Extra classrooms needed (rounded up; zero if population shrinks)."""
        pupils = self.additional_pupils(current, future)
        return max(0, math.ceil(pupils / self.pupils_per_classroom))
