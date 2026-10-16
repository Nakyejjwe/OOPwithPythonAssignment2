"""Mini-Project 4: rainfall regimes, crop suitability and regional similarity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import find_peaks

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class Region:
    """Monthly rainfall (mm, Jan–Dec) for one region."""

    def __init__(self, name: str, rainfall: Sequence[float]) -> None:
        arr = np.asarray(rainfall, dtype=float)
        if not name:
            raise ValueError("Region name must be non-empty")
        if arr.shape != (12,):
            raise ValueError(f"Expected 12 monthly values, got shape {arr.shape}")
        if np.any(arr < 0) or not np.all(np.isfinite(arr)):
            raise ValueError("Rainfall must be finite and non-negative")
        self.name = name
        self.rainfall = arr

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():.0f} mm)"

    def __len__(self) -> int:
        return 12

    def annual_total(self) -> float:
        return float(self.rainfall.sum())

    def mean(self) -> float:
        return float(self.rainfall.mean())

    def wettest_month(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    def cv(self) -> float:
        """Coefficient of variation of monthly rainfall (sample stdev / mean)."""
        m = self.mean()
        if m == 0:
            raise ZeroDivisionError("CV undefined for a region with no rain")
        return float(self.rainfall.std(ddof=1) / m)

    def rainy_seasons(self, prominence: float = 20.0) -> list[str]:
        """Peak months found by ``scipy.signal.find_peaks`` on the *circular* year.

        The series is padded with Dec before Jan and Jan after Dec so a peak in
        December or January is not missed at the array edge.
        """
        padded = np.concatenate([self.rainfall[-1:], self.rainfall, self.rainfall[:1]])
        idx, _ = find_peaks(padded, prominence=prominence)
        idx = [i - 1 for i in idx if 1 <= i <= 12]
        return [MONTHS[i] for i in sorted(idx)]

    def regime(self, prominence: float = 20.0) -> str:
        n = len(self.rainy_seasons(prominence))
        return {0: "no clear season", 1: "unimodal", 2: "bimodal"}.get(n, f"multimodal ({n} peaks)")


@dataclass(frozen=True)
class CropRule:
    """Suitable monthly rainfall band for a crop during its growing season (mm/month)."""

    crop: str
    min_mm: float
    max_mm: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.min_mm < 0 or self.min_mm >= self.max_mm:
            raise ValueError("Need 0 <= min_mm < max_mm")

    def classify(self, mm: float) -> str:
        if mm < 0:
            raise ValueError("Rainfall cannot be negative")
        if mm < self.min_mm:
            return "Drought risk"
        if mm > self.max_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop}"

    def score(self, mm: float) -> int:
        """-1 = too dry, 0 = suitable, +1 = too wet (for heatmaps)."""
        label = self.classify(mm)
        return -1 if label == "Drought risk" else (1 if label == "Waterlogging risk" else 0)


# Seasonal totals from FAO crop water information, spread over the main growth months.
DEFAULT_CROPS: tuple[CropRule, ...] = (
    CropRule("maize", 100, 200, "FAO: 500–800 mm per 4-month season → ≈125–200 mm/month"),
    CropRule("beans", 60, 150, "FAO: 300–500 mm per ~3-month season; sensitive to waterlogging"),
    CropRule("coffee", 100, 220, "FAO/UCDA: arabica/robusta ≈1,200–2,000 mm/yr, well distributed"),
)


# --------------------------------------------------------------------------- #
# Similarity / distance
# --------------------------------------------------------------------------- #
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """(a · b) / (‖a‖ ‖b‖). Measures the angle between vectors, not their size."""
    u, v = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if u.shape != v.shape:
        raise ValueError("Vectors must have the same shape")
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    return float(np.dot(u, v) / (nu * nv))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def pairwise_matrix(regions: Sequence[Region], metric) -> np.ndarray:
    n = len(regions)
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = metric(regions[i].rainfall, regions[j].rainfall)
    return M


def suitability_table(regions: Sequence[Region], crops: Sequence[CropRule]) -> dict[str, dict[str, list[str]]]:
    """{region: {crop: [label per month]}}."""
    return {r.name: {c.crop: [c.classify(v) for v in r.rainfall] for c in crops} for r in regions}
