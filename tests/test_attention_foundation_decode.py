"""CPU-safe correctness tests for the attention-foundation candidate.

The protected RTX 5070 Ti evaluator imports
``attention.foundation.attention_forward`` from main and from a candidate
checkout and compares them on the frozen benchmark. It decides the speedup;
these tests only pin what a contributor MUST hold regardless of device:

* the one-token decode path stays exact -- bit-for-bit the same as the default
  ``F.scaled_dot_product_attention`` main computes, and within the manifest's
  Frobenius/absolute tolerance of the fp32 oracle;
* prefill is untouched;
* the decode backend priority can never crash the candidate (MATH is always in
  the list) and never leaks the future in a causal prefill.

CPU only. Every SDPA backend except MATH is CUDA-only, so on this CI box the
priority context degrades to MATH -- exactly the guaranteed-fallback path the
implementation promises -- which is why exactness here proves the fallback is
sound even though it cannot prove the GPU speedup.

Run:  python tests/test_attention_foundation_decode.py
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
import torch.nn.functional as functional  # noqa: E402

from attention.foundation import attention_forward  # noqa: E402

# Manifest tolerances (benchmarks/attention-foundation-v1-rtx5070ti.json).
MAX_REL_FROBENIUS = 0.005
MAX_ABS = 0.05


def _oracle(q, k, v, *, causal):
    q32, k32, v32 = q.float(), k.float(), v.float()
    scores = torch.matmul(q32, k32.transpose(-1, -2)) / math.sqrt(float(q.shape[-1]))
    if causal:
        q_len, kv_len = q.shape[-2], k.shape[-2]
        mask = torch.ones((q_len, kv_len), dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v32)


def _rel_frobenius(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12))


# (label, q_shape, kv_shape, causal). Shapes mirror the manifest's decode/prefill
# families (q_len==1 decode, causal prefill, head_dim 64/128) at reduced kv/heads
# -- correctness is shape-independent, and modest sizes keep this stable on a
# CPU-only torch build. The protected evaluator uses the full manifest lengths.
DECODE_CASES = [
    ("decode-kv512-d128", (1, 4, 1, 128), (1, 4, 512, 128), False),
    ("decode-kv2048-d128", (1, 4, 1, 128), (1, 4, 2048, 128), False),
    ("decode-kv1024-d64", (1, 8, 1, 64), (1, 8, 1024, 64), False),
]
PREFILL_CASES = [
    ("prefill-n256-d128-causal", (1, 4, 256, 128), (1, 4, 256, 128), True),
    ("prefill-n128-d64-causal", (1, 8, 128, 64), (1, 8, 128, 64), True),
    ("prefill-n192-d64-noncausal", (1, 8, 192, 64), (1, 8, 192, 64), False),
]


def _mk(q_shape, kv_shape, *, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(q_shape, generator=gen)
    k = torch.randn(kv_shape, generator=gen)
    v = torch.randn(kv_shape, generator=gen)
    return q, k, v


@pytest.mark.parametrize("label,q_shape,kv_shape,causal", DECODE_CASES + PREFILL_CASES)
def test_matches_production_bitwise(label, q_shape, kv_shape, causal):
    # The candidate must return exactly what main's default SDPA returns; the
    # backend priority is a perf hint, never a numeric change.
    q, k, v = _mk(q_shape, kv_shape)
    got = attention_forward(q, k, v, causal=causal)
    production = functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=causal
    )
    assert torch.equal(got, production), label


@pytest.mark.parametrize("label,q_shape,kv_shape,causal", DECODE_CASES + PREFILL_CASES)
def test_within_oracle_tolerance(label, q_shape, kv_shape, causal):
    q, k, v = _mk(q_shape, kv_shape)
    got = attention_forward(q, k, v, causal=causal)
    oracle = _oracle(q, k, v, causal=causal)
    assert torch.isfinite(got).all(), label
    assert _rel_frobenius(got, oracle) < MAX_REL_FROBENIUS, label
    assert float((got.double() - oracle.double()).abs().max()) < MAX_ABS, label


def test_decode_shape_is_preserved():
    q, k, v = _mk((2, 8, 1, 128), (2, 8, 4096, 128))
    got = attention_forward(q, k, v, causal=False)
    assert tuple(got.shape) == (2, 8, 1, 128)


def test_noncontiguous_decode_is_exact():
    # The manifest's guard workloads feed strided (noncontiguous) tensors; a
    # backend that rejects them must fall through, not corrupt or crash.
    gen = torch.Generator().manual_seed(3)
    q = torch.randn((1, 8, 1, 128), generator=gen)[..., ::2]
    k = torch.randn((1, 8, 2048, 128), generator=gen)[..., ::2]
    v = torch.randn((1, 8, 2048, 128), generator=gen)[..., ::2]
    got = attention_forward(q, k, v, causal=False)
    production = functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    assert torch.equal(got, production)


def test_causal_prefill_does_not_leak_the_future():
    # Perturbing the second half of K/V must not move any first-half output.
    q, k, v = _mk((1, 8, 256, 64), (1, 8, 256, 64), seed=7)
    base = attention_forward(q, k, v, causal=True)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 128:, :] += 9.0
    v2[:, :, 128:, :] += 9.0
    perturbed = attention_forward(q, k2, v2, causal=True)
    delta = float((base[:, :, :128, :] - perturbed[:, :, :128, :]).abs().max())
    assert delta == 0.0, delta


def test_priority_list_keeps_math_as_guaranteed_fallback():
    # The never-crash guarantee rests on MATH always being in the priority list.
    from attention.foundation import _decode_backend_priority

    backends = _decode_backend_priority()
    if backends is None:
        pytest.skip("torch build without SDPBackend enum")
    from torch.nn.attention import SDPBackend

    assert SDPBackend.MATH in backends
    assert backends[-1] == SDPBackend.MATH


def test_prefill_takes_the_default_path(monkeypatch):
    # q_len > 1 must not even consult the decode priority helper.
    import attention.foundation as foundation

    called = {"n": 0}
    real = foundation._decode_backend_priority

    def spy():
        called["n"] += 1
        return real()

    monkeypatch.setattr(foundation, "_decode_backend_priority", spy)
    q, k, v = _mk((1, 8, 64, 64), (1, 8, 64, 64))
    attention_forward(q, k, v, causal=True)
    assert called["n"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
