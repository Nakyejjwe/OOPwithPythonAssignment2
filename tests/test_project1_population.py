import statistics

import numpy as np
import pytest

from src.forecasting import (CAGRForecaster, FibonacciRatioForecaster, LinearTrendForecaster,
                             bootstrap_interval, mae, mape, rmse)
from src.population import ClassroomPlanner, DistrictPopulation

YEARS = list(range(2015, 2025))
GULU = [320, 330, 345, 360, 375, 390, 410, 430, 455, 480]


def test_repr_len_and_validation():
    d = DistrictPopulation("Gulu", YEARS, GULU)
    assert len(d) == 10 and "Gulu" in repr(d)
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", YEARS, GULU[:-1])          # unequal lengths
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", YEARS, [-1] + GULU[1:])     # negative value


def test_variance_ddof_relationship():
    d = DistrictPopulation("Gulu", YEARS, GULU)
    n = len(d)
    assert d.stats_numpy(ddof=1)["variance"] == pytest.approx(statistics.variance(GULU))
    assert d.stats_numpy(ddof=0)["variance"] == pytest.approx(statistics.variance(GULU) * (n - 1) / n)


def test_cagr_hand_calculation():
    d = DistrictPopulation("Gulu", YEARS, GULU)
    assert d.cagr() == pytest.approx((480 / 320) ** (1 / 9) - 1)
    assert CAGRForecaster().fit([100, 121]).predict(1)[0] == pytest.approx(146.41)  # 121 × 1.21


def test_linear_trend_recovers_exact_line():
    y = 5 + 2 * np.arange(8)
    pred = LinearTrendForecaster().fit(y).predict(3)
    np.testing.assert_allclose(pred, [21, 23, 25])


def test_fibonacci_ratios():
    np.testing.assert_allclose(FibonacciRatioForecaster().ratios(3), [2, 1.5, 5 / 3])


def test_metrics_and_edge_cases():
    assert mae([1, 2], [2, 4]) == 1.5
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])                                  # zero actual
    with pytest.raises(ValueError):
        mae([], [])                                           # empty input
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(2)                    # predict before fit


def test_classroom_planner_rounding_and_decline():
    p = ClassroomPlanner()
    assert p.additional_classrooms(100, 101) == 4             # 180 pupils / 53 → 3.4 → 4
    assert p.additional_classrooms(100, 90) == 0              # shrinking population


def test_bootstrap_interval_contains_point():
    y = 100 + 3 * np.arange(10) + np.random.default_rng(0).normal(0, 2, 10)
    point, lo, hi = bootstrap_interval(LinearTrendForecaster, y, 3, n_boot=200)
    assert np.all(lo <= point) and np.all(point <= hi)
