"""Tests for eval.label -- pure logic against synthetic results, no GPU.

    python eval/tests/test_label.py        (or)   python -m pytest eval/tests -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.label import label, SIGNIFICANCE, BUCKETS


def _result(gated=False, improvement=True, score=1.0):
    return {"gated": gated, "improvement": improvement, "score": score}


def test_gated_is_reject():
    out = label(_result(gated=True, improvement=True, score=5.0), frontier_score=1.0, ref_anchor=1.0)
    assert out["verdict"] == "REJECT"


def test_non_dominant_is_none():
    out = label(_result(gated=False, improvement=False, score=0.0), frontier_score=1.0, ref_anchor=1.0)
    assert out["verdict"] == "none"


def test_first_admission_is_baseline():
    out = label(_result(score=1.0), frontier_score=0.0, ref_anchor=0.0)
    assert out["verdict"] == "BASELINE"


def test_below_significance_floor_is_none():
    # 1% gain over frontier -- below the 2% SIGNIFICANCE floor.
    frontier = 1.0
    out = label(_result(score=frontier * 1.01), frontier_score=frontier, ref_anchor=frontier)
    assert out["verdict"] == "none"
    assert out["delta_pct"] < SIGNIFICANCE * 100


def test_tier_sizing_uses_fixed_anchor_not_frontier():
    # ref_anchor fixed at 1.0 throughout. A late-game PR facing a much higher
    # frontier still gets a real tier if its ABSOLUTE gain (vs the fixed
    # anchor) is large, even though its RELATIVE gain vs the frontier is thin.
    ref_anchor = 1.0
    frontier = 10.0
    # +6% over frontier (clears significance) but delta=0.6 vs ref_anchor=1.0
    # -> tier_basis 0.6 -> should land in the L bucket (>=0.25).
    out = label(_result(score=frontier * 1.06), frontier_score=frontier, ref_anchor=ref_anchor)
    assert out["verdict"] == "L", out


def test_tiers_are_monotonic_with_gain():
    frontier = 1.0
    ref_anchor = 1.0
    seen = []
    for pct in (0.03, 0.06, 0.12, 0.30):
        out = label(_result(score=frontier * (1 + pct)), frontier_score=frontier, ref_anchor=ref_anchor)
        seen.append(out["verdict"])
    # Strictly increasing tier order for strictly increasing gains.
    order = {v: i for i, (_, v) in enumerate(reversed(BUCKETS))}
    ranks = [order[v] for v in seen]
    assert ranks == sorted(ranks), seen


def test_zero_ref_anchor_falls_back_to_significance_basis():
    # No prior admitted strategy to anchor against (ref_anchor=0) shouldn't
    # crash -- falls back to the frontier-relative significance as the basis.
    out = label(_result(score=1.10), frontier_score=1.0, ref_anchor=0.0)
    assert out["verdict"] in {v for _, v in BUCKETS}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
