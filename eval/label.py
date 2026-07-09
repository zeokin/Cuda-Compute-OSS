"""Deterministic PR verdict labels for the matmul-strategy scoring pipeline.

``label()`` turns one ``evaluate()``-style transform result into a single
verdict -- REJECT / none / BASELINE / S..L -- as a pure function of the
measurements, so an independent re-run of the same PR always converges on
the same verdict.

This module does NOT recompute the accuracy/dominance gates -- those already
live in :mod:`eval.metrics` and are applied once by ``evaluate()`` (the
``gated``/``improvement``/``score`` fields on each transform's result). This
module only tiers what already passed that gate, against the current
frontier and a fixed reference anchor.
"""
from __future__ import annotations

# Significance floor: a gain must clear this fraction of the CURRENT frontier
# score to count as verified at all. Sub-floor gains are labeled "none" --
# never aggregated across runs to manufacture a headline (BENCHMARKS.md).
SIGNIFICANCE = 0.02

# Tier thresholds -- fraction of gain over a FIXED reference anchor, not the
# moving frontier, so a tier keeps the same meaning as the frontier moves
# (mirrors sparkinfer's DIFF_REF anti-drift design in bench/scripts/label.py).
# Ordered highest-threshold-first; the first bucket a gain clears wins.
BUCKETS = (
    (0.25, "L"),
    (0.10, "M"),
    (SIGNIFICANCE, "S"),
)


def label(result: dict, frontier_score: float, ref_anchor: float) -> dict:
    """Return a verdict dict for one transform's ``evaluate()`` result.

    result         : one entry from ``evaluate()['transforms'][name]``. Only
                      three fields are read -- ``gated`` (accuracy below the
                      floor), ``improvement`` (dominates the exact baseline on
                      every cost axis, per ``metrics.dominates_exact``), and
                      ``score`` (the perf_score, already zeroed unless
                      ``improvement`` is true).
    frontier_score : current best admitted ``score`` on this track. ``<= 0``
                      means nothing has been admitted yet on this track.
    ref_anchor     : a fixed ``score`` that tier width is measured against
                      (e.g. the first-ever admitted score on this track), so
                      tier meaning does not drift as the frontier moves.

    Returns {"verdict": str, "reason": str | None, "delta_pct": float | None}.
    """
    if result.get("gated"):
        return {"verdict": "REJECT", "reason": "accuracy below the floor",
                "delta_pct": None}

    if not result.get("improvement"):
        return {"verdict": "none",
                "reason": "does not dominate exact on every cost axis",
                "delta_pct": None}

    score = result["score"]

    if frontier_score <= 0:
        return {"verdict": "BASELINE",
                "reason": "first admitted strategy on this track",
                "delta_pct": None}

    delta = score - frontier_score
    significance = delta / frontier_score
    delta_pct = round(significance * 100, 2)

    if significance <= SIGNIFICANCE:
        return {"verdict": "none",
                "reason": f"gain {significance:.1%} does not clear the "
                          f"{SIGNIFICANCE:.0%} significance floor",
                "delta_pct": delta_pct}

    tier_basis = delta / ref_anchor if ref_anchor > 0 else significance
    verdict = next((v for threshold, v in BUCKETS if tier_basis >= threshold), "S")
    return {"verdict": verdict, "reason": None, "delta_pct": delta_pct}
