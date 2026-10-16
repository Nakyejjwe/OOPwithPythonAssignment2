"""Shared forecasting toolkit used by Mini-Projects 1 and 5.

Every model inherits from the abstract :class:`Forecaster`, which fixes the
interface ``fit(y) -> self`` and ``predict(horizon) -> np.ndarray``. Error
metrics and a residual-bootstrap for prediction intervals live here too, so
no project duplicates them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Error metrics
# --------------------------------------------------------------------------- #
def _pair(actual: Sequence[float], predicted: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"Shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if a.size == 0:
        raise ValueError("Cannot compute an error metric on empty arrays")
    return a, p


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error (same units as the data)."""
    a, p = _pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error (same units as the data; punishes big misses)."""
    a, p = _pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error, in percent. Undefined if any actual is 0."""
    a, p = _pair(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / a)) * 100)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Forecaster(ABC):
    """Abstract one-dimensional time-series forecaster.

    Subclasses implement :meth:`_fit` and :meth:`_predict`; this class handles
    input validation and the "must fit before predict" rule.
    """

    #: Short label used in tables and plot legends.
    name: str = "base"

    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    # public API ----------------------------------------------------------- #
    def fit(self, y: Sequence[float]) -> "Forecaster":
        """Fit the model to a series and return ``self`` (allows chaining)."""
        arr = np.asarray(y, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Series must be one-dimensional")
        if arr.size < self.min_length:
            raise ValueError(f"{self.name} needs at least {self.min_length} observations, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Series contains NaN or infinite values")
        self._y = arr
        self._fit(arr)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` values."""
        if self._y is None:
            raise RuntimeError("Call fit() before predict()")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return self._predict(int(horizon))

    def fitted(self) -> np.ndarray:
        """In-sample fitted values (NaN where the model has no fitted value)."""
        if self._y is None:
            raise RuntimeError("Call fit() before fitted()")
        return self._fitted()

    @property
    def min_length(self) -> int:
        """Smallest series length the model can be fitted on."""
        return 2

    def __repr__(self) -> str:
        state = "fitted" if self._y is not None else "unfitted"
        return f"{type(self).__name__}({state})"

    # hooks for subclasses ------------------------------------------------- #
    @abstractmethod
    def _fit(self, y: np.ndarray) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...

    @abstractmethod
    def _fitted(self) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Concrete models
# --------------------------------------------------------------------------- #
class LinearTrendForecaster(Forecaster):
    """Straight-line trend y = a + b·t fitted with ``np.polyfit`` (degree 1)."""

    name = "Linear trend"

    def _fit(self, y: np.ndarray) -> None:
        self._t = np.arange(y.size)
        self.slope, self.intercept = np.polyfit(self._t, y, deg=1)

    def _predict(self, horizon: int) -> np.ndarray:
        t_future = np.arange(self._y.size, self._y.size + horizon)
        return self.intercept + self.slope * t_future

    def _fitted(self) -> np.ndarray:
        return self.intercept + self.slope * self._t


class CAGRForecaster(Forecaster):
    """Exponential growth at the series' compound annual growth rate.

    CAGR = (last / first) ** (1 / (n - 1)) - 1, then y_{T+k} = y_T (1 + CAGR)^k.
    """

    name = "Exponential (CAGR)"

    def _fit(self, y: np.ndarray) -> None:
        if y[0] <= 0 or y[-1] <= 0:
            raise ValueError("CAGR needs strictly positive first and last values")
        self.rate = (y[-1] / y[0]) ** (1 / (y.size - 1)) - 1

    def _predict(self, horizon: int) -> np.ndarray:
        k = np.arange(1, horizon + 1)
        return self._y[-1] * (1 + self.rate) ** k

    def _fitted(self) -> np.ndarray:
        return self._y[0] * (1 + self.rate) ** np.arange(self._y.size)


def fibonacci(n: int) -> np.ndarray:
    """First ``n`` Fibonacci numbers starting 1, 1, 2, 3, ... (integer dtype)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    out = np.zeros(n, dtype=np.int64)
    a, b = 1, 1
    for i in range(n):
        out[i] = a
        a, b = b, a + b
    return out


class FibonacciRatioForecaster(Forecaster):
    """The previous cohort's model: scale the last value by successive F(k+1)/F(k).

    Step k of the forecast multiplies the previous value by ``F(start+k)/F(start+k-1)``.
    With ``start=1`` the multipliers are 2, 1.5, 1.667, 1.6, ... → φ ≈ 1.618, i.e.
    the series grows ~60 % per period whatever the data say. The model ignores
    the training data except for its last value — kept here only so it can be
    benchmarked and critiqued.
    """

    name = "Fibonacci ratio"

    def __init__(self, start: int = 1) -> None:
        super().__init__()
        if start < 1:
            raise ValueError("start must be >= 1")
        self.start = start

    @property
    def min_length(self) -> int:
        return 1

    def ratios(self, horizon: int) -> np.ndarray:
        """The multiplicative factors applied at each forecast step."""
        f = fibonacci(self.start + horizon + 1).astype(float)  # F1..F(start+h+1)
        return f[self.start + 1:self.start + 1 + horizon] / f[self.start:self.start + horizon]

    def _fit(self, y: np.ndarray) -> None:  # nothing to estimate
        pass

    def _predict(self, horizon: int) -> np.ndarray:
        return self._y[-1] * np.cumprod(self.ratios(horizon))

    def _fitted(self) -> np.ndarray:
        # One-step-ahead "fit": each value = previous actual × first ratio.
        out = np.full(self._y.size, np.nan)
        out[1:] = self._y[:-1] * self.ratios(1)[0]
        return out


class MovingAverageForecaster(Forecaster):
    """k-period simple moving average; the forecast is flat at the last window mean."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.name = f"{window}-day moving average"

    @property
    def min_length(self) -> int:
        return self.window

    def _fit(self, y: np.ndarray) -> None:
        self.level = float(np.mean(y[-self.window:]))

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)

    def _fitted(self) -> np.ndarray:
        out = np.full(self._y.size, np.nan)
        for t in range(self.window, self._y.size):
            out[t] = self._y[t - self.window:t].mean()
        return out


class SimpleExpSmoothingForecaster(Forecaster):
    """Simple exponential smoothing: ℓ_t = α y_t + (1-α) ℓ_{t-1}, ℓ_0 = y_0."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        self.alpha = alpha
        self.name = f"SES (α={alpha:.2f})"

    @property
    def min_length(self) -> int:
        return 1

    def _fit(self, y: np.ndarray) -> None:
        levels = np.empty(y.size)
        levels[0] = y[0]
        for t in range(1, y.size):
            levels[t] = self.alpha * y[t] + (1 - self.alpha) * levels[t - 1]
        self._levels = levels

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._levels[-1])

    def _fitted(self) -> np.ndarray:
        out = np.full(self._y.size, np.nan)
        out[1:] = self._levels[:-1]  # one-step-ahead forecasts
        return out


class SeasonalNaiveForecaster(Forecaster):
    """Forecast = value observed one season ago (default season = 7 days)."""

    def __init__(self, season: int = 7) -> None:
        super().__init__()
        if season < 1:
            raise ValueError("season must be >= 1")
        self.season = season
        self.name = f"Seasonal naïve (m={season})"

    @property
    def min_length(self) -> int:
        return self.season

    def _fit(self, y: np.ndarray) -> None:
        pass

    def _predict(self, horizon: int) -> np.ndarray:
        last_season = self._y[-self.season:]
        return np.array([last_season[k % self.season] for k in range(horizon)])

    def _fitted(self) -> np.ndarray:
        out = np.full(self._y.size, np.nan)
        out[self.season:] = self._y[:-self.season]
        return out


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def holdout_scores(model: Forecaster, train: Sequence[float], test: Sequence[float]) -> dict[str, float]:
    """Fit on ``train``, forecast ``len(test)`` steps, return MAE/RMSE/MAPE."""
    pred = model.fit(train).predict(len(test))
    return {"MAE": mae(test, pred), "RMSE": rmse(test, pred), "MAPE": mape(test, pred)}


def walk_forward(model: Forecaster, y: Sequence[float], first_target: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-origin one-step-ahead backtest.

    For every index t >= ``first_target`` the model is refitted on y[:t] and
    asked for one step. Returns (actuals, forecasts).
    """
    arr = np.asarray(y, dtype=float)
    if not 1 <= first_target < arr.size:
        raise ValueError("first_target must be inside the series")
    preds = np.array([model.fit(arr[:t]).predict(1)[0] for t in range(first_target, arr.size)])
    return arr[first_target:], preds


def bootstrap_interval(
    model_factory,
    y: Sequence[float],
    horizon: int,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual-bootstrap prediction interval.

    1. Fit the model, take in-sample residuals.
    2. Repeat ``n_boot`` times: rebuild a synthetic history (fitted + resampled
       residuals), refit, forecast, and add resampled residuals to the forecast
       (so both parameter and noise uncertainty are reflected).
    3. Return (point forecast, lower, upper) percentiles.

    ``model_factory`` is a zero-argument callable returning a fresh model.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    arr = np.asarray(y, dtype=float)
    base = model_factory().fit(arr)
    fitted = base.fitted()
    mask = ~np.isnan(fitted)
    resid = arr[mask] - fitted[mask]
    resid = resid - resid.mean()
    point = base.predict(horizon)
    sims = np.empty((n_boot, horizon))
    for b in range(n_boot):
        synthetic = arr.copy()
        synthetic[mask] = fitted[mask] + rng.choice(resid, size=mask.sum(), replace=True)
        try:
            path = model_factory().fit(synthetic).predict(horizon)
        except ValueError:
            path = point
        sims[b] = path + rng.choice(resid, size=horizon, replace=True)
    alpha = (1 - level) / 2
    lower, upper = np.quantile(sims, [alpha, 1 - alpha], axis=0)
    return point, lower, upper
