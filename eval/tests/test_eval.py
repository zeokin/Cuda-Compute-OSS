"""Tests for the eval system.

Metric tests run anywhere (pure NumPy). End-to-end evaluate/scaling tests use
the GPU (PyTorch) and skip when no CUDA/MPS device is present.

    python eval/tests/test_eval.py        (or)   python -m pytest eval/tests -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval import metrics
from eval.evaluator import EvalConfig, effective_rank_m, evaluate, estimate_scaling
from strategy.subspace import default_rank_m


class _Skip(Exception):
    pass


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()
                    or (getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()))
    except Exception:  # noqa: BLE001
        return False


HAVE_GPU = _gpu_available()

# The metric tests below are pure NumPy and run anywhere. The end-to-end GPU
# tests are marked so pytest skips them *cleanly* (not as errors) when no device
# is present; the internal ``_Skip`` guard keeps the __main__ runner working when
# these are executed directly with `python eval/tests/test_eval.py`.
try:
    import pytest
    _gpu_only = pytest.mark.skipif(
        not HAVE_GPU, reason="no CUDA/MPS GPU; CCO computes on GPU only")
except ImportError:
    def _gpu_only(fn):
        return fn


# ---- metrics -------------------------------------------------------------
def test_accuracy_identical_is_one():
    C = np.random.default_rng(0).standard_normal((32, 32))
    assert metrics.accuracy(C, C) == 1.0


def test_accuracy_is_bounded_zero_one():
    rng = np.random.default_rng(1)
    C = rng.standard_normal((32, 32))
    Chat = C + 100.0 * rng.standard_normal((32, 32))   # very wrong
    a = metrics.accuracy(C, Chat)
    assert 0.0 <= a <= 1.0


def test_accuracy_floors_at_zero():
    # Approx with the negated matrix -> error >> 1 -> clamped to 0, never negative.
    C = np.ones((8, 8))
    assert metrics.accuracy(C, -C) == 0.0


def test_rel_frobenius_error_keeps_climbing_past_the_accuracy_clamp():
    # accuracy = max(0, 1 - error) floors at 0 once error >= 1, but the real
    # relative Frobenius error keeps growing -- the two are not interchangeable
    # above the clamp, so a caller wanting the true error (e.g. eval.evaluator's
    # report) must call rel_frobenius_error directly rather than back-deriving
    # it from accuracy (which would flatten every error >= 1 to exactly 1).
    rng = np.random.default_rng(2)
    C = rng.standard_normal((16, 16))
    Chat = -3.0 * C                       # ||C - Chat||_F / ||C||_F == 4.0 exactly
    err = metrics.rel_frobenius_error(C, Chat)
    acc = metrics.accuracy(C, Chat)
    assert abs(err - 4.0) < 1e-9
    assert acc == 0.0


def test_score_gated_by_accuracy_floor():
    # Accuracy below the floor -> score forced to 0 regardless of speed/memory.
    gated = metrics.score(0.5, peak_vram_bytes=1e6, latency_s=0.01,
                          accuracy_floor=0.9)
    assert gated == 0.0
    ok = metrics.score(0.95, peak_vram_bytes=1e6, latency_s=0.01,
                       accuracy_floor=0.9)
    assert ok > 0.0


def test_score_no_floor_by_default():
    # Default floor is 0.0 -> even low accuracy is scored (not gated).
    assert metrics.score(0.01, 1e6, 0.01) > 0.0


def test_score_monotonic_in_accuracy():
    lo = metrics.score(0.2, 1e6, 0.01)
    hi = metrics.score(0.9, 1e6, 0.01)
    assert hi > lo


def test_default_rank_m_matches_strategy_floor():
    # Strategy floors M at 64; eval must report the same default, not bare N//8.
    assert default_rank_m(256) == 64
    assert default_rank_m(256) != 256 // 8
    assert default_rank_m(12000) == 1500
    assert default_rank_m(32) == 32


def test_effective_rank_m_uses_strategy_default():
    assert effective_rank_m(EvalConfig(n=256, rank_m=None)) == 64
    assert effective_rank_m(EvalConfig(n=256, rank_m=48)) == 48


# ---- dominance gate (the improvement rule) -------------------------------
def test_dominance_all_axes_below_exact():
    # Faster, lighter, fewer FLOPs than exact -> admitted as an improvement.
    assert metrics.dominates_exact(
        latency_s=0.5, peak_vram_bytes=1e6, flop_ratio_vs_exact=4.0,
        exact_latency_s=1.0, exact_peak_vram_bytes=2e6) is True


def test_dominance_rejects_slower_than_exact():
    # Accurate and light, but slower than exact -> not an improvement.
    assert metrics.dominates_exact(
        latency_s=1.5, peak_vram_bytes=1e6, flop_ratio_vs_exact=4.0,
        exact_latency_s=1.0, exact_peak_vram_bytes=2e6) is False


def test_dominance_rejects_heavier_than_exact():
    # Faster and fewer FLOPs, but uses more VRAM than exact -> not an improvement.
    assert metrics.dominates_exact(
        latency_s=0.5, peak_vram_bytes=3e6, flop_ratio_vs_exact=4.0,
        exact_latency_s=1.0, exact_peak_vram_bytes=2e6) is False


def test_dominance_rejects_no_flop_win():
    # Faster and lighter, but does not reduce FLOP count -> not an improvement.
    assert metrics.dominates_exact(
        latency_s=0.5, peak_vram_bytes=1e6, flop_ratio_vs_exact=1.0,
        exact_latency_s=1.0, exact_peak_vram_bytes=2e6) is False


# ---- end-to-end evaluate (GPU) -------------------------------------------
@_gpu_only
def test_evaluate_smoke():
    if not HAVE_GPU:
        raise _Skip()
    ev = EvalConfig(n=96, pairs=2, dtype="fp32", fill="lowrank", data_rank=4,
                    transforms=["rsvd"], verbose=False)
    out = evaluate(ev)
    assert out["config"]["rank_m"] == default_rank_m(96)
    assert set(out["transforms"]) == {"rsvd"}
    for r in out["transforms"].values():
        assert 0.0 <= r["accuracy"] <= 1.0
        assert r["latency_s"] > 0.0
        assert r["score"] >= 0.0
    assert out["best"] == "rsvd"


@_gpu_only
def test_rsvd_accurate_on_lowrank():
    # On genuinely low-rank data the data-aware rsvd basis reconstructs closely.
    if not HAVE_GPU:
        raise _Skip()
    ev = EvalConfig(n=128, pairs=2, dtype="fp32", fill="lowrank", data_rank=6,
                    rank_m=48, transforms=["rsvd"], verbose=False)
    out = evaluate(ev)
    assert out["transforms"]["rsvd"]["accuracy"] > 0.99


@_gpu_only
def test_scaling_exponent_runs():
    if not HAVE_GPU:
        raise _Skip()
    ev = EvalConfig(dtype="fp32", fill="lowrank", data_rank=4, rank_m=16,
                    transforms=["rsvd"], verbose=False)
    out = estimate_scaling([64, 128, 192], ev)
    assert "fitted_exponent_p" in out
    assert np.isfinite(out["fitted_exponent_p"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = skipped = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except _Skip:
            skipped += 1
            print(f"SKIP  {fn.__name__} (no GPU)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed - skipped}/{len(fns) - skipped} passed"
          + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)
