import numpy as np
import pytest

from src.fishery import FishStock, PriceModel, RiskAssessor, revenue_stats, weekly_revenue


def test_one_step_hand_calculation():
    N, H = FishStock(h=0.1).simulate(1)
    # 4000 + 0.4*4000*(1-0.4) - 400 = 4560
    assert N[1] == pytest.approx(4560) and H[0] == pytest.approx(400)


def test_msy_and_equilibrium():
    s = FishStock(h=0.2)
    assert s.msy == 1000 and s.equilibrium_stock() == 5000 and s.sustainable_yield() == pytest.approx(1000)
    assert FishStock(h=0.5).equilibrium_stock() == 0          # h >= r collapses


def test_price_bounds_and_reproducibility():
    p1 = PriceModel(seed=1).simulate(200, n_paths=50)
    p2 = PriceModel(seed=1).simulate(200, n_paths=50)
    assert p1.min() >= 9000 and p1.max() <= 16000
    np.testing.assert_array_equal(p1, p2)


def test_cv_is_unit_free():
    rev = np.array([10.0, 12.0, 9.0, 11.0])
    assert revenue_stats(rev)["cv"] == pytest.approx(revenue_stats(rev * 1e6)["cv"])
    assert revenue_stats(rev * 1e6)["variance"] == pytest.approx(revenue_stats(rev)["variance"] * 1e12)


def test_risk_classes_and_invalid():
    r = RiskAssessor()
    assert [r.classify(c) for c in (0.05, 0.2, 0.5)] == ["Low", "Medium", "High"]
    with pytest.raises(ValueError):
        r.classify(-0.1)
    with pytest.raises(ValueError):
        RiskAssessor.value_at_risk([])


def test_closed_season_zero_harvest():
    _, H = FishStock(closed_weeks=range(8)).simulate(52)
    assert np.all(H[:8] == 0) and np.all(H[8:] > 0)


def test_revenue_units():
    assert weekly_revenue(np.array([2.0]), np.array([12000.0]))[0] == 24_000_000
