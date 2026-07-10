"""CPU-safe tests for the local attention playground."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

from attention import AttentionSpec

if torch is not None:
    from attention import (
        adaptive_hybrid_attention,
        adaptive_spectral_global_mix,
        correlation_hybrid_attention,
        correlation_spectral_global_mix,
        exact_attention,
        generate_qkv,
        hybrid_attention,
        landmark_global_attention,
        landmark_hybrid_attention,
        local_window_attention,
        spectral_global_mix,
    )


def _skip_if_no_torch():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


def _sample(seq=16, dim=8):
    q = torch.randn(1, 2, seq, dim, dtype=torch.float32)
    k = torch.randn(1, 2, seq, dim, dtype=torch.float32)
    v = torch.randn(1, 2, seq, dim, dtype=torch.float32)
    return q, k, v


def test_attention_spec_defaults_are_stable():
    spec = AttentionSpec()
    assert spec.batch == 1
    assert spec.heads == 8
    assert spec.seq == 4096
    assert spec.window == 256


def test_attention_spec_rejects_invalid_dimensions():
    for kwargs in (
        {"batch": 0},
        {"heads": 0},
        {"seq": 0},
        {"dim": 0},
        {"window": -1},
    ):
        try:
            AttentionSpec(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AttentionSpec({kwargs!r}) should raise ValueError")


def test_attention_spec_rejects_invalid_branch_weights():
    for kwargs in (
        {"local_weight": -0.1},
        {"global_weight": -0.1},
        {"local_weight": 0.0, "global_weight": 0.0},
        {"freq_decay": -0.1},
    ):
        try:
            AttentionSpec(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AttentionSpec({kwargs!r}) should raise ValueError")


def test_generate_qkv_uses_spec_shape():
    if _skip_if_no_torch():
        return
    spec = AttentionSpec(batch=2, heads=3, seq=10, dim=7, dtype="fp32", device="cpu")
    q, k, v = generate_qkv(spec)
    assert tuple(q.shape) == (2, 3, 10, 7)
    assert tuple(k.shape) == (2, 3, 10, 7)
    assert tuple(v.shape) == (2, 3, 10, 7)


def test_exact_attention_shape():
    if _skip_if_no_torch():
        return
    q, k, v = _sample()
    out = exact_attention(q, k, v)
    assert tuple(out.shape) == tuple(v.shape)


def test_local_window_matches_exact_when_window_covers_sequence():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=12, dim=4)
    exact = exact_attention(q, k, v)
    local = local_window_attention(q, k, v, window=12, block_size=5)
    assert torch.allclose(local, exact, atol=1e-5, rtol=1e-5)


def test_spectral_global_mix_preserves_shape_and_finiteness():
    if _skip_if_no_torch():
        return
    _q, _k, v = _sample(seq=18, dim=6)
    out = spectral_global_mix(v, freq_decay=0.5)
    assert tuple(out.shape) == tuple(v.shape)
    assert torch.isfinite(out).all()


def test_spectral_global_mix_handles_fp16():
    # fp16 is the default benchmark dtype; torch.fft has no half support, so the
    # mixer must upcast (like its siblings) instead of crashing.
    if _skip_if_no_torch():
        return
    v = torch.randn(1, 2, 16, 8, dtype=torch.float16)
    out = spectral_global_mix(v, freq_decay=0.5)
    assert out.dtype == torch.float16
    assert torch.isfinite(out.float()).all()


def test_hybrid_attention_handles_fp16():
    if _skip_if_no_torch():
        return
    q = torch.randn(1, 2, 16, 8, dtype=torch.float16)
    k = torch.randn(1, 2, 16, 8, dtype=torch.float16)
    v = torch.randn(1, 2, 16, 8, dtype=torch.float16)
    out = hybrid_attention(q, k, v, window=4)
    assert torch.isfinite(out.float()).all()


def test_adaptive_spectral_global_mix_preserves_shape_and_finiteness():
    if _skip_if_no_torch():
        return
    q, _k, v = _sample(seq=18, dim=6)
    out = adaptive_spectral_global_mix(q, v, freq_decay=0.5, gate_strength=0.2)
    assert tuple(out.shape) == tuple(v.shape)
    assert torch.isfinite(out).all()


def test_correlation_spectral_global_mix_preserves_shape_and_finiteness():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=18, dim=6)
    out = correlation_spectral_global_mix(q, k, v, temperature=1.0, freq_decay=0.1)
    assert tuple(out.shape) == tuple(v.shape)
    assert torch.isfinite(out).all()


def test_landmark_global_attention_preserves_shape_and_finiteness():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=18, dim=6)
    out = landmark_global_attention(q, k, v, num_landmarks=6)
    assert tuple(out.shape) == tuple(v.shape)
    assert torch.isfinite(out).all()


def test_landmark_global_attention_causal_is_finite_for_early_queries():
    # Early queries can precede every landmark position, masking their whole
    # softmax row to -inf. That must not produce NaN.
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=20, dim=8)
    for policy in ("pooled", "topk"):
        out = landmark_global_attention(
            q, k, v, num_landmarks=5, causal=True, policy=policy
        )
        assert torch.isfinite(out).all(), f"non-finite output for policy={policy}"


def test_landmark_hybrid_causal_is_finite():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=20, dim=8)
    out = landmark_hybrid_attention(
        q, k, v, window=4, causal=True, num_landmarks=5
    )
    assert torch.isfinite(out).all()


def test_landmark_global_attention_matches_exact_with_one_landmark_per_token():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=12, dim=4)
    exact = exact_attention(q, k, v)
    landmark = landmark_global_attention(q, k, v, num_landmarks=12)
    assert torch.allclose(landmark, exact, atol=1e-5, rtol=1e-5)


def test_topk_landmark_global_attention_matches_exact_with_one_landmark_per_token():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=12, dim=4)
    exact = exact_attention(q, k, v)
    landmark = landmark_global_attention(q, k, v, num_landmarks=12, policy="topk")
    assert torch.allclose(landmark, exact, atol=1e-5, rtol=1e-5)


def test_hybrid_equals_local_when_global_weight_is_zero():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=14, dim=5)
    local = local_window_attention(q, k, v, window=3, block_size=4)
    hybrid = hybrid_attention(
        q, k, v,
        window=3,
        block_size=4,
        local_weight=1.0,
        global_weight=0.0,
    )
    assert torch.allclose(hybrid, local, atol=1e-5, rtol=1e-5)


def test_adaptive_hybrid_equals_local_when_global_weight_is_zero():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=14, dim=5)
    local = local_window_attention(q, k, v, window=3, block_size=4)
    hybrid = adaptive_hybrid_attention(
        q, k, v,
        window=3,
        block_size=4,
        local_weight=1.0,
        global_weight=0.0,
    )
    assert torch.allclose(hybrid, local, atol=1e-5, rtol=1e-5)


def test_correlation_hybrid_equals_local_when_global_weight_is_zero():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=14, dim=5)
    local = local_window_attention(q, k, v, window=3, block_size=4)
    hybrid = correlation_hybrid_attention(
        q, k, v,
        window=3,
        block_size=4,
        local_weight=1.0,
        global_weight=0.0,
    )
    assert torch.allclose(hybrid, local, atol=1e-5, rtol=1e-5)


def test_landmark_hybrid_equals_local_when_global_weight_is_zero():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=14, dim=5)
    local = local_window_attention(q, k, v, window=3, block_size=4)
    hybrid = landmark_hybrid_attention(
        q, k, v,
        window=3,
        block_size=4,
        local_weight=1.0,
        global_weight=0.0,
        num_landmarks=4,
    )
    assert torch.allclose(hybrid, local, atol=1e-5, rtol=1e-5)


def test_topk_landmark_hybrid_equals_local_when_global_weight_is_zero():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=14, dim=5)
    local = local_window_attention(q, k, v, window=3, block_size=4)
    hybrid = landmark_hybrid_attention(
        q, k, v,
        window=3,
        block_size=4,
        local_weight=1.0,
        global_weight=0.0,
        num_landmarks=4,
        landmark_policy="topk",
    )
    assert torch.allclose(hybrid, local, atol=1e-5, rtol=1e-5)


def test_attention_benchmark_can_compare_both_modes():
    if _skip_if_no_torch():
        return
    from attention.benchmark import run_once

    result = run_once(
        batch=1,
        heads=1,
        seq=8,
        dim=4,
        dtype="fp32",
        window=3,
        mode="both",
        device="cpu",
    )
    assert set(result["candidates"]) == {"fixed", "adaptive"}
    assert "exact" in result
    assert "quality" in result["candidates"]["fixed"]
    assert "quality" in result["candidates"]["adaptive"]


def test_attention_benchmark_can_compare_all_modes():
    if _skip_if_no_torch():
        return
    from attention.benchmark import run_once

    result = run_once(
        batch=1,
        heads=1,
        seq=8,
        dim=4,
        dtype="fp32",
        window=3,
        mode="all",
        landmarks=4,
        device="cpu",
    )
    assert set(result["candidates"]) == {"fixed", "adaptive", "corrfft", "landmark", "topk"}
    assert "quality" in result["candidates"]["landmark"]
    assert "quality" in result["candidates"]["topk"]
    assert "quality" in result["candidates"]["corrfft"]


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
