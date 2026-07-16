"""Regression tests: causal correlation mixer must not leak future q/k (issue TBD).

#154 made the V-convolution causal (and tested only v-perturbation), but the lag
kernel was still built from whole-sequence Q/K statistics, so out[t] depended on
future q/k. A single shared kernel cannot be causal, so causal runs fall back to
the causal spectral low-pass. Non-causal correlation is unchanged.

CPU-safe: skips cleanly when torch is not installed.
Run:  python tests/test_correlation_causal_qk.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

if torch is not None:
    from attention.hybrid import correlation_spectral_global_mix, spectral_global_mix


def _skip():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


def _worst_past_change(which, seed, seq=12):
    torch.manual_seed(seed)
    q, k, v = (torch.randn(1, 1, seq, 3, dtype=torch.float64) for _ in range(3))
    base = correlation_spectral_global_mix(q, k, v, causal=True)
    pert = {"q": q, "k": k, "v": v}[which].clone()
    pert[0, 0, -1] += 50.0
    args = {"q": q, "k": k, "v": v}
    args[which] = pert
    out = correlation_spectral_global_mix(args["q"], args["k"], args["v"], causal=True)
    return max((out[0, 0, t] - base[0, 0, t]).abs().max().item() for t in range(seq - 1))


def test_causal_does_not_leak_future_q_or_k():
    if _skip():
        return
    for which in ("q", "k"):
        worst = max(_worst_past_change(which, s) for s in range(20))
        assert worst < 1e-9, f"future {which} leaked into a past output: {worst}"


def test_causal_still_does_not_leak_future_v():
    """Keeps the #154 property."""
    if _skip():
        return
    worst = max(_worst_past_change("v", s) for s in range(20))
    assert worst < 1e-9


def test_causal_matches_causal_spectral_lowpass():
    if _skip():
        return
    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 1, 20, 5, dtype=torch.float64) for _ in range(3))
    got = correlation_spectral_global_mix(q, k, v, causal=True, freq_decay=1.0)
    ref = spectral_global_mix(v, freq_decay=1.0, causal=True)
    assert torch.allclose(got, ref, atol=1e-9)


def test_noncausal_correlation_is_unchanged():
    """Non-causal still uses the Q/K lag kernel (distinct from a plain low-pass)."""
    if _skip():
        return
    torch.manual_seed(4)
    q, k, v = (torch.randn(1, 1, 20, 5, dtype=torch.float64) for _ in range(3))
    corr = correlation_spectral_global_mix(q, k, v, causal=False)
    plain = spectral_global_mix(v, causal=False)
    assert torch.isfinite(corr).all() and not torch.allclose(corr, plain, atol=1e-4)


def test_causal_path_is_only_the_early_spectral_fallback():
    """#260: after the early ``if causal: return spectral_global_mix(...)``,
    later ``if causal:`` bodies were unreachable dead code. Pin that the
    function source has exactly one causal branch -- the early return.
    """
    if _skip():
        return
    import inspect
    import re

    src = inspect.getsource(correlation_spectral_global_mix)
    hits = re.findall(r"^\s*if causal\s*:", src, flags=re.MULTILINE)
    assert len(hits) == 1, (
        f"expected a single causal branch (early spectral fallback), found {len(hits)}"
    )
    assert "return spectral_global_mix" in src


if __name__ == "__main__":
    fns = [v for kk, v in sorted(globals().items()) if kk.startswith("test_")]
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
