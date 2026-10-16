"""Mini-Project 3: Lake Victoria fish stock, price and revenue-risk models."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


class FishStock:
    """Discrete logistic growth with proportional harvesting.

        N(t+1) = N(t) + r N(t) (1 − N(t)/K) − h N(t)

    Stock is in tonnes and one step is one week. ``closed_weeks`` lists week
    indices (0–51, repeating every year) in which no harvesting happens.
    """

    def __init__(
        self,
        r: float = 0.4,
        K: float = 10_000.0,
        N0: float = 4_000.0,
        h: float = 0.1,
        closed_weeks: Iterable[int] = (),
    ) -> None:
        if r <= 0:
            raise ValueError("Growth rate r must be positive")
        if K <= 0:
            raise ValueError("Carrying capacity K must be positive")
        if N0 < 0:
            raise ValueError("Initial stock cannot be negative")
        if not 0 <= h <= 1:
            raise ValueError("Harvest rate h must be between 0 and 1")
        self.r, self.K, self.N0, self.h = float(r), float(K), float(N0), float(h)
        self.closed_weeks = frozenset(int(w) % 52 for w in closed_weeks)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f}, N0={self.N0:,.0f}, h={self.h})"

    def simulate(self, weeks: int = 52) -> tuple[np.ndarray, np.ndarray]:
        """Return (stock, harvest): stock has ``weeks + 1`` values, harvest has ``weeks``."""
        if weeks < 0:
            raise ValueError("weeks must be non-negative")
        N = np.empty(weeks + 1)
        H = np.empty(weeks)
        N[0] = self.N0
        for t in range(weeks):
            h_t = 0.0 if (t % 52) in self.closed_weeks else self.h
            H[t] = h_t * N[t]
            N[t + 1] = max(0.0, N[t] + self.r * N[t] * (1 - N[t] / self.K) - H[t])
        return N, H

    # theory ------------------------------------------------------------------ #
    @property
    def msy(self) -> float:
        """Maximum sustainable yield rK/4 (tonnes per week), reached at h = r/2."""
        return self.r * self.K / 4

    def equilibrium_stock(self) -> float:
        """Long-run stock K(1 − h/r); zero if h ≥ r (fishery collapses)."""
        return max(0.0, self.K * (1 - self.h / self.r))

    def sustainable_yield(self) -> float:
        """Long-run weekly harvest h · N* at this harvest rate."""
        return self.h * self.equilibrium_stock()


class PriceModel:
    """Weekly fish price (UGX/kg) as a seeded, bounded Gaussian random walk."""

    def __init__(
        self,
        start: float = 12_000.0,
        step_sd: float = 400.0,
        lower: float = 9_000.0,
        upper: float = 16_000.0,
        seed: int = 7,
    ) -> None:
        if not lower < upper:
            raise ValueError("lower bound must be below upper bound")
        if not lower <= start <= upper:
            raise ValueError("start price must lie within the bounds")
        if step_sd < 0:
            raise ValueError("step_sd cannot be negative")
        self.start, self.step_sd = float(start), float(step_sd)
        self.lower, self.upper = float(lower), float(upper)
        self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"PriceModel(start={self.start:,.0f}, sd={self.step_sd:,.0f}, bounds=[{self.lower:,.0f}, {self.upper:,.0f}])"

    def simulate(self, weeks: int = 52, n_paths: int = 1) -> np.ndarray:
        """Return prices of shape (n_paths, weeks). Bounds applied by clipping each step."""
        if weeks <= 0 or n_paths <= 0:
            raise ValueError("weeks and n_paths must be positive")
        shocks = self.rng.normal(0, self.step_sd, size=(n_paths, weeks))
        prices = np.empty((n_paths, weeks))
        p = np.full(n_paths, self.start)
        for t in range(weeks):
            prices[:, t] = p
            p = np.clip(p + shocks[:, t], self.lower, self.upper)
        return prices


def weekly_revenue(harvest_tonnes: np.ndarray, price_ugx_per_kg: np.ndarray) -> np.ndarray:
    """Revenue in UGX = harvest (t) × 1000 kg/t × price (UGX/kg). Broadcasts over paths."""
    return np.asarray(harvest_tonnes) * 1000.0 * np.asarray(price_ugx_per_kg)


def revenue_stats(revenue: Sequence[float]) -> dict[str, float]:
    """Mean, median, sample variance, stdev and CV via the ``statistics`` module."""
    data = [float(v) for v in revenue]
    if len(data) < 2:
        raise ValueError("Need at least two revenue values")
    mean = statistics.mean(data)
    sd = statistics.stdev(data)
    return {
        "mean": mean,
        "median": statistics.median(data),
        "variance": statistics.variance(data),
        "stdev": sd,
        "cv": sd / mean if mean != 0 else float("inf"),
    }


@dataclass
class RiskAssessor:
    """Classifies revenue risk from the coefficient of variation (CV = σ/μ).

    CV is unit-free, so the same thresholds work whether revenue is in UGX,
    USD or millions. Default bands: CV < 0.15 Low, 0.15–0.30 Medium, ≥ 0.30 High
    — roughly "a typical week is within ±15 % of average" for Low.
    """

    low: float = 0.15
    high: float = 0.30
    labels: tuple[str, str, str] = field(default=("Low", "Medium", "High"))

    def __post_init__(self) -> None:
        if not 0 <= self.low < self.high:
            raise ValueError("Need 0 <= low < high")

    def classify(self, cv: float) -> str:
        if cv < 0 or np.isnan(cv):
            raise ValueError("CV must be a non-negative number")
        if cv < self.low:
            return self.labels[0]
        if cv < self.high:
            return self.labels[1]
        return self.labels[2]

    def assess(self, revenue: Sequence[float]) -> tuple[float, str]:
        cv = revenue_stats(revenue)["cv"]
        return cv, self.classify(cv)

    @staticmethod
    def value_at_risk(annual_revenues: Sequence[float], alpha: float = 0.05) -> dict[str, float]:
        """Historical-simulation VaR of annual revenue.

        ``quantile`` is the revenue that is exceeded with probability 1 − α
        (the "bad year"); ``var`` is the shortfall from the mean in that year.
        """
        arr = np.asarray(annual_revenues, dtype=float)
        if arr.size == 0:
            raise ValueError("No simulated revenues supplied")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        q = float(np.quantile(arr, alpha))
        return {"quantile": q, "mean": float(arr.mean()), "var": float(arr.mean() - q)}
