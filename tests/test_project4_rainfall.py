import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine_distance

from src.rainfall import CropRule, Region, cosine_similarity, euclidean

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


def test_region_summary():
    r = Region("Kampala", KAMPALA)
    assert r.annual_total() == 1600 and r.wettest_month() == "May" and r.driest_month() == "Sep"


def test_cosine_matches_scipy():
    assert cosine_similarity(KAMPALA, MBARARA) == pytest.approx(1 - scipy_cosine_distance(KAMPALA, MBARARA))


def test_cosine_ignores_scale_but_euclidean_does_not():
    a = np.array(KAMPALA, float)
    assert cosine_similarity(a, 3 * a) == pytest.approx(1.0)
    assert euclidean(a, 3 * a) > 0


def test_zero_vector_and_bad_length():
    with pytest.raises(ValueError):
        cosine_similarity([0] * 12, KAMPALA)
    with pytest.raises(ValueError):
        Region("Short", [10] * 11)


def test_crop_rule_boundaries():
    rule = CropRule("maize", 100, 200)
    assert rule.classify(99) == "Drought risk"
    assert rule.classify(100) == "Good for maize"
    assert rule.classify(201) == "Waterlogging risk"
    with pytest.raises(ValueError):
        CropRule("bad", 200, 100)


def test_bimodal_detection():
    assert Region("Mbarara", MBARARA).regime() == "bimodal"
