import numpy as np
import pytest

from src.forecasting import MovingAverageForecaster, SeasonalNaiveForecaster, SimpleExpSmoothingForecaster, walk_forward
from src.taxi import FleetPlanner, LinearMarket, Route

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


def test_route_revenue():
    r = Route("Ntinda", NTINDA, 2000)
    assert r.total_revenue() == sum(NTINDA) * 2000 and len(r) == 10


def test_equilibrium_hand_solution():
    P, Q = LinearMarket(120, 0.02, 10, 0.03).equilibrium()
    assert P == pytest.approx(2200) and Q == pytest.approx(76)


def test_moving_average_and_ses_limits():
    assert MovingAverageForecaster(3).fit(NTINDA).predict(1)[0] == pytest.approx((52 + 47 + 45) / 3)
    assert SimpleExpSmoothingForecaster(1.0).fit(NTINDA).predict(1)[0] == 45   # α=1 → naïve


def test_walk_forward_length():
    actual, pred = walk_forward(MovingAverageForecaster(3), NTINDA, first_target=3)
    assert len(actual) == len(pred) == 7


def test_fleet_planner():
    f = FleetPlanner()
    assert f.capacity_per_vehicle == 112
    assert f.vehicles(112) == 2      # 112*1.15 = 128.8 → 2 vehicles
    assert f.vehicles(0) == 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Route("x", [], 2000)
    with pytest.raises(ValueError):
        SimpleExpSmoothingForecaster(0)
    with pytest.raises(ValueError):
        FleetPlanner().vehicles(-3)


def test_seasonal_naive_repeats_last_week():
    y = np.arange(14)
    np.testing.assert_array_equal(SeasonalNaiveForecaster(7).fit(y).predict(9), [7, 8, 9, 10, 11, 12, 13, 7, 8])
