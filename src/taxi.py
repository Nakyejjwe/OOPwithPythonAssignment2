"""Mini-Project 5: matatu routes, fare equilibrium and fleet planning."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg


class Route:
    """A taxi route: daily passenger counts and a flat fare (UGX)."""

    def __init__(self, name: str, passengers: Sequence[float], fare: float) -> None:
        arr = np.asarray(passengers, dtype=float)
        if not name:
            raise ValueError("Route name must be non-empty")
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("Passenger counts must be a non-empty 1-D sequence")
        if np.any(arr < 0):
            raise ValueError("Passenger counts cannot be negative")
        if fare <= 0:
            raise ValueError("Fare must be positive")
        self.name = name
        self.passengers = arr
        self.fare = float(fare)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare:,.0f})"

    def __len__(self) -> int:
        return int(self.passengers.size)

    def daily_revenue(self) -> np.ndarray:
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def stats(self) -> dict[str, float]:
        data = self.passengers.tolist()
        out = {"mean": statistics.mean(data)}
        if len(data) >= 2:
            out["variance"] = statistics.variance(data)
            out["stdev"] = statistics.stdev(data)
        return out


@dataclass(frozen=True)
class LinearMarket:
    """Linear demand Qd = a − bP and supply Qs = c + dP (b, d > 0)."""

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        if self.b <= 0 or self.d <= 0:
            raise ValueError("Demand must slope down and supply up (b, d > 0)")

    def equilibrium(self) -> tuple[float, float]:
        """Solve  [1, b; 1, -d] [Q, P]ᵀ = [a, c]ᵀ  with scipy.linalg.solve."""
        A = np.array([[1.0, self.b], [1.0, -self.d]])
        rhs = np.array([self.a, self.c])
        Q, P = linalg.solve(A, rhs)
        return float(P), float(Q)

    def demand(self, P: float) -> float:
        return self.a - self.b * P

    def supply(self, P: float) -> float:
        return self.c + self.d * P


@dataclass(frozen=True)
class FleetPlanner:
    """Vehicles needed = ceil(demand × (1 + buffer) / (trips × seats))."""

    trips_per_day: int = 8
    seats: int = 14
    buffer: float = 0.15

    def __post_init__(self) -> None:
        if self.trips_per_day <= 0 or self.seats <= 0:
            raise ValueError("trips_per_day and seats must be positive")
        if self.buffer < 0:
            raise ValueError("buffer cannot be negative")

    @property
    def capacity_per_vehicle(self) -> int:
        return self.trips_per_day * self.seats

    def vehicles(self, passengers: float) -> int:
        if passengers < 0:
            raise ValueError("Passenger forecast cannot be negative")
        if passengers == 0:
            return 0
        return math.ceil(passengers * (1 + self.buffer) / self.capacity_per_vehicle)


def simulate_weekly_passengers(days: int = 60, base: float = 50, seed: int = 11) -> np.ndarray:
    """Synthetic daily counts with a weekly pattern (Fri busy, Sun quiet) + noise.

    Day 0 is a Monday.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    profile = np.array([1.00, 0.95, 0.98, 1.05, 1.35, 1.10, 0.60])  # Mon..Sun
    pattern = profile[np.arange(days) % 7]
    counts = base * pattern + rng.normal(0, 3, days)
    return np.clip(np.round(counts), 0, None)
