"""Regression tests: adaptive global branch must be causal in causal mode (issue #181).

adaptive_spectral_global_mix applied a symmetric zero-phase FFT low-pass gated by a
summary of all queries, so adaptive_hybrid_attention(causal=True) leaked the future.
A zero-phase adaptive gate cannot be causal cheaply, so causal runs fall back to the
causal spectral low-pass. Non-causal adaptive is unchanged.

CPU-safe: skips cleanly when torch is not installed.
Run:  python tests/test_adaptive_causal.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

if torch is not None:
    from attention.hybrid import (
        adaptive_hybrid_attention,
        adaptive_spectral_global_mix,
        spectral_global_mix,
    )


def _skip():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


def test_adaptive_hybrid_causal_does_not_leak_future():
    if _skip():
        return
    for which in ("q", "v"):
        for seed in range(20):
            torch.manual_seed(seed)
            q, k, v = (torch.randn(1, 1, 16, 4) for _ in range(3))
            base = adaptive_hybrid_attention(q, k, v, window=4, causal=True)
            q2, v2 = q.clone(), v.clone()
            (q2 if which == "q" else v2)[0, 0, 15] += 8.0
            out2 = adaptive_hybrid_attention(q2, k, v2, window=4, causal=True)
            worst = max((out2[0, 0, t] - base[0, 0, t]).abs().max().item() for t in range(15))
            assert worst < 1e-5, f"future {which} leaked into a past query (seed {seed}): {worst}"


def test_adaptive_causal_matches_causal_spectral_lowpass():
    if _skip():
        return
    torch.manual_seed(3)
    q, v = (torch.randn(1, 1, 20, 5) for _ in range(2))
    got = adaptive_spectral_global_mix(q, v, freq_decay=0.7, gate_strength=0.3, causal=True)
    ref = spectral_global_mix(v, freq_decay=0.7, causal=True)
    assert torch.allclose(got, ref, atol=1e-6)


def test_adaptive_causal_is_finite():
    if _skip():
        return
    torch.manual_seed(1)
    q, v = (torch.randn(1, 1, 16, 4) for _ in range(2))
    out = adaptive_spectral_global_mix(q, v, causal=True)
    assert out.shape == v.shape and torch.isfinite(out.float()).all()


def test_adaptive_noncausal_is_unchanged():
    """Non-causal adaptive still uses its q-adaptive gate (distinct from spectral)."""
    if _skip():
        return
    torch.manual_seed(4)
    q, v = (torch.randn(1, 1, 20, 5) for _ in range(2))
    adaptive = adaptive_spectral_global_mix(q, v, gate_strength=0.5, causal=False)
    plain = spectral_global_mix(v, causal=False)
    assert torch.isfinite(adaptive).all() and not torch.allclose(adaptive, plain, atol=1e-4)


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
