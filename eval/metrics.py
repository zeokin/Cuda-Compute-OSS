"""Scoring metrics for the smart (subspace) strategy vs the normal (exact) product.

Accuracy is the **Relative Frobenius Norm Error** folded into a bounded [0, 1]
score so it can be multiplied into a performance formula without ever going
negative or blowing up:

    Accuracy = max(0, 1 - ||C - Ĉ||_F / ||C||_F)

where C is the normal (exact) product and Ĉ is the smart approximation.

The final score rewards accurate, memory-light, low-latency strategies:

    score = Accuracy × (1 / Peak_VRAM) × (1 / Latency)

and is hard-gated to zero unless the strategy is admitted as an **improvement**
over the exact baseline. Admission (the dominance rule in BENCHMARKS.md) requires
accuracy at/above the floor AND every cost axis — latency, peak VRAM, and FLOP
count — strictly below the exact baseline. A strategy that is fast and tiny but
wrong, or accurate but slower / heavier than exact, therefore cannot win.
"""
from __future__ import annotations

import numpy as np

GIB = 1024.0 ** 3
MIB = 1024.0 ** 2
_VRAM_UNITS = {"bytes": 1.0, "mib": MIB, "gib": GIB}


def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def rel_frobenius_error(C_exact, C_approx) -> float:
    """||C - Ĉ||_F / ||C||_F  (0 = identical, ~1 = uncorrelated, can exceed 1)."""
    C = _f64(C_exact)
    diff = _f64(C_approx) - C
    denom = float(np.linalg.norm(C)) or 1.0
    return float(np.linalg.norm(diff) / denom)


def accuracy(C_exact, C_approx) -> float:
    """Bounded accuracy score in [0, 1]: max(0, 1 - relative Frobenius error)."""
    return max(0.0, 1.0 - rel_frobenius_error(C_exact, C_approx))


def score(
    accuracy_score: float,
    peak_vram_bytes: float,
    latency_s: float,
    accuracy_floor: float = 0.0,
    vram_unit: str = "gib",
) -> float:
    """score = accuracy × (1/Peak_VRAM) × (1/Latency), gated by an accuracy floor.

    Peak_VRAM is expressed in ``vram_unit`` (default GiB) so the number stays in
    a human-readable range; it is a *relative* ranking metric across strategies
    measured under the same units. Returns 0 when ``accuracy_score`` is below
    ``accuracy_floor`` (default 0.0 -> never gated).
    """
    if accuracy_score < accuracy_floor:
        return 0.0
    unit = _VRAM_UNITS[vram_unit]
    vram = max(peak_vram_bytes, 1.0) / unit          # guard against 0 / negative
    lat = max(latency_s, 1e-9)                        # guard against 0 latency
    return float(accuracy_score * (1.0 / vram) * (1.0 / lat))


def dominates_exact(
    latency_s: float,
    peak_vram_bytes: float,
    flop_ratio_vs_exact: float | None,
    exact_latency_s: float,
    exact_peak_vram_bytes: float,
) -> bool:
    """The dominance gate (BENCHMARKS.md): does the strategy reduce *every* cost
    axis versus the exact baseline?

    Returns True only if wall-clock latency, peak VRAM, and FLOP count are all
    strictly lower than exact (``flop_ratio_vs_exact`` > 1 means fewer FLOPs than
    exact). A single regressing axis disqualifies the strategy — we never average
    a win on one axis against a loss on another. Accuracy is gated separately by
    the floor in :func:`score`.
    """
    return (
        latency_s < exact_latency_s
        and peak_vram_bytes < exact_peak_vram_bytes
        and (flop_ratio_vs_exact or 0.0) > 1.0
    )
