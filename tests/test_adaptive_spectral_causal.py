"""Regression tests: the adaptive spectral global branch must be causal in causal mode.

`adaptive_spectral_global_mix` previously ignored `causal` entirely: it low-passed V
with a zero-phase rfft (mixing the whole sequence) and gated with a mean of Q over
every query. So an `adaptive` hybrid run with `causal=True` let `out[t]` depend on
`q[t+1:]` / `v[t+1:]` -- the same class of leak fixed for the spectral, correlation,
and topk branches. The causal branch now uses the EMA linear convolution for V and a
running (cumulative) mean of Q for the gate, so only the past is visible.

CPU-safe: skips cleanly when torch is not installed.
Run:  python tests/test_adaptive_spectral_causal.py
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
    )


def _skip():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


def test_adaptive_mix_causal_is_finite():
    if _skip():
        return
    torch.manual_seed(0)
    q, v = (torch.randn(1, 1, 16, 4) for _ in range(2))
    out = adaptive_spectral_global_mix(q, v, causal=True)
    assert out.shape == v.shape and torch.isfinite(out).all()


def test_adaptive_mix_does_not_leak_future():
    """Perturbing a future q/v must not change any earlier position's output."""
    if _skip():
        return
    for which in ("q", "v"):
        for seed in range(20):
            torch.manual_seed(seed)
            q, v = (torch.randn(1, 1, 16, 4) for _ in range(2))
            base = adaptive_spectral_global_mix(q, v, causal=True)
            q2, v2 = q.clone(), v.clone()
            (q2 if which == "q" else v2)[0, 0, 15] += 8.0
            out2 = adaptive_spectral_global_mix(q2, v2, causal=True)
            worst = max((out2[0, 0, t] - base[0, 0, t]).abs().max().item() for t in range(15))
            assert worst < 1e-5, f"future {which} leaked into a past position (seed {seed}): {worst}"


def test_adaptive_hybrid_causal_does_not_leak_future():
    """End-to-end: adaptive_hybrid_attention with causal=True must not leak the future."""
    if _skip():
        return
    for which in ("q", "k", "v"):
        for seed in range(10):
            torch.manual_seed(seed)
            q, k, v = (torch.randn(1, 1, 16, 4) for _ in range(3))
            base = adaptive_hybrid_attention(q, k, v, window=2, causal=True)
            pert = [q.clone(), k.clone(), v.clone()]
            pert[{"q": 0, "k": 1, "v": 2}[which]][0, 0, 15] += 8.0
            out2 = adaptive_hybrid_attention(*pert, window=2, causal=True)
            worst = max((out2[0, 0, t] - base[0, 0, t]).abs().max().item() for t in range(15))
            assert worst < 1e-5, f"future {which} leaked (seed {seed}): {worst}"


def test_adaptive_mix_freq_decay_zero_is_running_gated_v():
    """At freq_decay=0 the causal low-pass is identity, so out == v scaled by the gate."""
    if _skip():
        return
    torch.manual_seed(2)
    q, v = (torch.randn(1, 1, 12, 3) for _ in range(2))
    out = adaptive_spectral_global_mix(q, v, freq_decay=0.0, gate_strength=0.25, causal=True)
    counts = torch.arange(1, 13, dtype=v.dtype).view(1, 1, -1, 1)
    gate = 1.0 + 0.25 * torch.tanh(torch.cumsum(q, dim=-2) / counts)
    assert torch.allclose(out, v * gate, atol=1e-5)


def test_adaptive_mix_noncausal_is_unchanged():
    """Non-causal path is untouched: still the zero-phase rfft low-pass + Q-mean gate."""
    if _skip():
        return
    torch.manual_seed(4)
    q, v = (torch.randn(1, 1, 20, 5) for _ in range(2))
    out = adaptive_spectral_global_mix(q, v, causal=False)
    vf = torch.fft.rfft(v.to(torch.float32), dim=-2)
    freqs = torch.arange(vf.shape[-2], dtype=torch.float32)
    base_gain = 1.0 / (1.0 + 1.0 * freqs)
    q_summary = torch.tanh(q.to(torch.float32).mean(dim=-2, keepdim=True))
    gate = base_gain.view(1, 1, -1, 1) * (1.0 + 0.25 * q_summary)
    expect = torch.fft.irfft(vf * gate.to(vf.dtype), n=20, dim=-2)
    assert torch.allclose(out, expect, atol=1e-5)


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
