"""Mini-Project 2: solar micro-grid dispatch as a linear system.

Base system (x = solar kWh, y = battery kWh):
    3x + 2y = D1   (daytime load)
    4x +  y = D2   (critical-equipment load)
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize


class MicroGrid:
    """A dispatch system ``A @ sources = demand``.

    Parameters
    ----------
    coefficients : square coefficient matrix A (default: the Kasese 2×2 system).
    source_names : label for each energy source (column of A).
    unit_costs : cost per kWh of each source in UGX.
    """

    DEFAULT_A = np.array([[3.0, 2.0], [4.0, 1.0]])

    def __init__(
        self,
        coefficients: Sequence[Sequence[float]] | None = None,
        source_names: Sequence[str] = ("solar", "battery"),
        unit_costs: Sequence[float] = (150.0, 450.0),
    ) -> None:
        A = np.array(self.DEFAULT_A if coefficients is None else coefficients, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Coefficient matrix must be square")
        if len(source_names) != A.shape[1] or len(unit_costs) != A.shape[1]:
            raise ValueError("Need one name and one unit cost per source")
        if any(c < 0 for c in unit_costs):
            raise ValueError("Unit costs cannot be negative")
        self.A = A
        self.source_names = tuple(source_names)
        self.unit_costs = np.asarray(unit_costs, dtype=float)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_sources={self.n}, det={self.determinant():.3g})"

    @property
    def n(self) -> int:
        return self.A.shape[0]

    # well-posedness -------------------------------------------------------- #
    def determinant(self) -> float:
        return float(np.linalg.det(self.A))

    def condition_number(self) -> float:
        return float(np.linalg.cond(self.A))

    def is_well_posed(self, tol: float = 1e-10, max_cond: float = 1e8) -> bool:
        """Unique solution exists (det ≠ 0) and the system is not numerically ill-conditioned."""
        return abs(self.determinant()) > tol and self.condition_number() < max_cond

    def _check(self) -> None:
        if not self.is_well_posed():
            raise np.linalg.LinAlgError(
                f"System is singular or ill-conditioned (det={self.determinant():.3g}, "
                f"cond={self.condition_number():.3g})")

    # solving --------------------------------------------------------------- #
    def solve_day(self, *demands: float) -> np.ndarray:
        """Solve for one day's source usage given one demand per constraint."""
        if len(demands) != self.n:
            raise ValueError(f"Expected {self.n} demand values, got {len(demands)}")
        b = np.asarray(demands, dtype=float)
        if np.any(b < 0):
            raise ValueError("Demand cannot be negative")
        self._check()
        return linalg.solve(self.A, b)

    def solve_loop(self, demand: np.ndarray) -> np.ndarray:
        """Solve each day separately. ``demand`` has shape (n, days); returns (n, days)."""
        demand = self._as_rhs(demand)
        self._check()
        return np.column_stack([linalg.solve(self.A, demand[:, j]) for j in range(demand.shape[1])])

    def solve_vectorised(self, demand: np.ndarray) -> np.ndarray:
        """Solve all days in one call by passing an (n, days) right-hand side."""
        demand = self._as_rhs(demand)
        self._check()
        return linalg.solve(self.A, demand)

    def _as_rhs(self, demand: np.ndarray) -> np.ndarray:
        D = np.asarray(demand, dtype=float)
        if D.ndim != 2 or D.shape[0] != self.n:
            raise ValueError(f"Demand must have shape ({self.n}, days)")
        if D.shape[1] == 0:
            raise ValueError("Demand has no days")
        return D

    # feasibility ------------------------------------------------------------ #
    @staticmethod
    def infeasible_days(solution: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """Boolean mask of days where any source would have to be negative."""
        return np.any(np.asarray(solution) < -tol, axis=0)

    def solve_nonnegative(self, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact solve where feasible; NNLS (closest non-negative fit) elsewhere.

        Returns (solution, residual_norm_per_day). Residual is 0 on feasible days
        and measures unmet/over-supplied demand on infeasible ones.
        """
        D = self._as_rhs(demand)
        sol = self.solve_vectorised(D)
        resid = np.zeros(D.shape[1])
        for j in np.where(self.infeasible_days(sol))[0]:
            sol[:, j], resid[j] = optimize.nnls(self.A, D[:, j])
        return sol, resid

    # costs -------------------------------------------------------------------- #
    def daily_cost(self, solution: np.ndarray) -> np.ndarray:
        """UGX cost per day: unit_costs · usage."""
        return self.unit_costs @ np.asarray(solution, dtype=float)


class HybridMicroGrid(MicroGrid):
    """Adds a diesel generator (z) and a third constraint (evening load).

        3x + 2y + 1z = D1
        4x + 1y + 2z = D2
        0x + 3y + 2z = D3   (evening load: no sun, battery + diesel)
    """

    DEFAULT_A = np.array([[3.0, 2.0, 1.0], [4.0, 1.0, 2.0], [0.0, 3.0, 2.0]])

    def __init__(
        self,
        coefficients: Sequence[Sequence[float]] | None = None,
        source_names: Sequence[str] = ("solar", "battery", "diesel"),
        unit_costs: Sequence[float] = (150.0, 450.0, 1200.0),
    ) -> None:
        super().__init__(coefficients if coefficients is not None else self.DEFAULT_A,
                         source_names, unit_costs)


# ---------------------------------------------------------------------------- #
# Input helpers
# ---------------------------------------------------------------------------- #
def prompt_positive_float(
    message: str,
    input_func: Callable[[str], str] = input,
    max_attempts: int = 5,
) -> float:
    """Ask until the user types a non-negative number. ``input_func`` is injectable for tests."""
    for _ in range(max_attempts):
        raw = input_func(message).strip()
        if not raw:
            print("  ✗ Empty input — please type a number.")
            continue
        try:
            value = float(raw)
        except ValueError:
            print(f"  ✗ '{raw}' is not a number.")
            continue
        if not np.isfinite(value) or value < 0:
            print("  ✗ Demand must be a non-negative, finite number.")
            continue
        return value
    raise ValueError(f"No valid input after {max_attempts} attempts")


def generate_demand_csv(path: str, days: int = 30, seed: int = 2026) -> pd.DataFrame:
    """Synthesise ``days`` of demand with a weekly cycle and noise; save to CSV.

    D1 (daytime load) ≈ 60 kWh, higher on weekdays (outpatient clinics);
    D2 (critical load) ≈ 1.1 × D1 plus noise — chosen so most days are feasible
    but a few are not, which exercises the infeasibility handling.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    day = np.arange(days)
    weekday = day % 7 < 5
    d1 = 60 + 6 * weekday - 4 * (~weekday) + 2.5 * np.sin(2 * np.pi * day / 7) + rng.normal(0, 3, days)
    d2 = 1.1 * d1 + rng.normal(0, 4, days)
    df = pd.DataFrame({
        "day": day + 1,
        "date": pd.date_range("2026-09-01", periods=days, freq="D").strftime("%Y-%m-%d"),
        "D1_kWh": d1.round(2),
        "D2_kWh": d2.round(2),
    })
    df.to_csv(path, index=False)
    return df


def load_demand_csv(path: str) -> np.ndarray:
    """Read the CSV and return a (2, days) demand matrix, validating values."""
    df = pd.read_csv(path)
    for col in ("D1_kWh", "D2_kWh"):
        if col not in df:
            raise ValueError(f"CSV missing column {col}")
    D = df[["D1_kWh", "D2_kWh"]].to_numpy(dtype=float).T
    if np.isnan(D).any() or (D < 0).any():
        raise ValueError("CSV contains missing or negative demand")
    return D
