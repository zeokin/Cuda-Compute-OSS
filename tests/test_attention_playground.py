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


def test_attention_spec_rejects_non_integer_fields():
    # window=2.5 used to construct successfully (only `window < 0` was
    # checked) and then crash later inside local_window_attention's tensor
    # slicing -- but only for (seq, window, block_size) combinations where a
    # block boundary lands on the non-integer value, so it passed for some
    # configs and crashed for others with a raw TypeError far from the real
    # cause. It and the other int-typed fields must be rejected at
    # construction instead. bool is also rejected even though it's a technical
    # int subclass (isinstance(True, int) is True) -- never a meaningful value
    # here.
    for kwargs in (
        {"batch": 1.5},
        {"heads": 2.5},
        {"seq": 10.0},
        {"dim": 8.5},
        {"window": 2.5},
        {"seed": 1.5},
        {"batch": True},
        {"window": False},
    ):
        try:
            AttentionSpec(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AttentionSpec({kwargs!r}) should raise ValueError")


def test_attention_spec_rejects_invalid_dtype():
    # dtype used to construct successfully and only fail later, deep in
    # data.torch_dtype, as an opaque KeyError.
    for bad in ("bf16", "float32", "int8", "", "FP16"):
        try:
            AttentionSpec(dtype=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AttentionSpec(dtype={bad!r}) should raise ValueError")
    for good in ("fp16", "fp32", "fp64"):
        AttentionSpec(dtype=good)  # must not raise


def test_attention_spec_rejects_invalid_device():
    # device used to construct successfully and only fail later, deep in
    # resolve_device (or not at all, for a syntactically-valid-but-wrong
    # string), well after construction and after generate_qkv had already
    # started building tensors.
    for bad in ("gpu", "cuda:abc", "tpu", "cuda:", "", "Cuda:0"):
        try:
            AttentionSpec(device=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AttentionSpec(device={bad!r}) should raise ValueError")
    for good in ("auto", "cpu", "mps", "cuda", "cuda:0", "cuda:7"):
        AttentionSpec(device=good)  # must not raise


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


def test_adaptive_spectral_global_mix_preserves_shape_and_finiteness():
    if _skip_if_no_torch():
        return
    q, _k, v = _sample(seq=18, dim=6)
    out = adaptive_spectral_global_mix(q, v, freq_decay=0.5, gate_strength=0.2)
    assert tuple(out.shape) == tuple(v.shape)
    assert torch.isfinite(out).all()


def test_spectral_global_mix_causal_does_not_see_the_future():
    """Regression for #164: the default (non-causal) branch is a symmetric FFT
    low-pass that mixes every position; causal=True must not read the future."""
    if _skip_if_no_torch():
        return
    torch.manual_seed(0)
    seq = 32
    v = torch.randn(1, 2, seq, 4, dtype=torch.float64)
    bumped = v.clone()
    bumped[:, :, -1, :] += 100.0                 # only the future-most token
    for freq_decay in (0.0, 0.7, 3.0):
        base = spectral_global_mix(v, freq_decay=freq_decay, causal=True)
        moved = spectral_global_mix(bumped, freq_decay=freq_decay, causal=True)
        assert (moved - base)[:, :, :-1, :].abs().max() < 1e-9, freq_decay


def test_spectral_global_mix_causal_freq_decay_zero_is_identity():
    """freq_decay=0 -> EMA pole a=1 -> identity, matching the non-causal gain==1."""
    if _skip_if_no_torch():
        return
    torch.manual_seed(1)
    v = torch.randn(1, 2, 16, 4, dtype=torch.float64)
    out = spectral_global_mix(v, freq_decay=0.0, causal=True)
    assert torch.allclose(out, v, atol=1e-12)


def test_spectral_global_mix_causal_preserves_constant_v():
    """Each causal output is a convex combination of the visible past, so a
    constant V comes back unchanged (the zero-padding must not shrink it)."""
    if _skip_if_no_torch():
        return
    torch.manual_seed(2)
    v = torch.full((1, 2, 20, 4), 2.5, dtype=torch.float64)
    out = spectral_global_mix(v, freq_decay=1.0, causal=True)
    assert (out - 2.5).abs().max() < 1e-9


def test_spectral_global_mix_noncausal_unchanged_by_new_flag():
    """The default branch must be bit-identical to the pre-flag behavior."""
    if _skip_if_no_torch():
        return
    torch.manual_seed(3)
    v = torch.randn(1, 2, 24, 5, dtype=torch.float64)
    for freq_decay in (0.0, 0.3, 1.0, 5.0):
        a = spectral_global_mix(v, freq_decay=freq_decay)              # default causal=False
        b = spectral_global_mix(v, freq_decay=freq_decay, causal=False)
        assert torch.equal(a, b)


def test_hybrid_attention_causal_does_not_see_the_future():
    """The leak must not survive through the default (fixed) hybrid wrapper (#164)."""
    if _skip_if_no_torch():
        return
    torch.manual_seed(4)
    seq, window = 32, 4
    q = torch.randn(1, 2, seq, 4, dtype=torch.float64)
    k = torch.randn(1, 2, seq, 4, dtype=torch.float64)
    v = torch.randn(1, 2, seq, 4, dtype=torch.float64)
    bumped = v.clone()
    bumped[:, :, -1, :] += 100.0
    base = hybrid_attention(q, k, v, window=window, causal=True)
    moved = hybrid_attention(q, k, bumped, window=window, causal=True)
    # v[-1] can reach out[t] via the local branch only for t >= seq-1-window.
    horizon = seq - 1 - window
    assert (moved - base)[:, :, :horizon, :].abs().max() < 1e-9


def _corr_sample(seq, dim=4, seed=0):
    torch.manual_seed(seed)
    return tuple(torch.randn(1, 2, seq, dim, dtype=torch.float64) for _ in range(3))


def test_adaptive_spectral_global_mix_causal_does_not_see_the_future():
    """Regression for #180: the default adaptive branch used a global q.mean over
    the full sequence and a symmetric FFT low-pass on V -- both leak the future."""
    if _skip_if_no_torch():
        return
    torch.manual_seed(10)
    seq = 32
    q, k, v = _corr_sample(seq, seed=10)
    bumped = v.clone()
    bumped[:, :, -1, :] += 100.0
    for freq_decay in (0.0, 0.7, 3.0):
        base = adaptive_spectral_global_mix(
            q, v, freq_decay=freq_decay, gate_strength=0.25, causal=True
        )
        moved = adaptive_spectral_global_mix(
            q, bumped, freq_decay=freq_decay, gate_strength=0.25, causal=True
        )
        assert (moved - base)[:, :, :-1, :].abs().max() < 1e-9, freq_decay


def test_adaptive_spectral_global_mix_causal_q_gate_is_prefix_only():
    """The adaptive gate must not read future Q either."""
    if _skip_if_no_torch():
        return
    torch.manual_seed(11)
    seq = 24
    q, k, v = _corr_sample(seq, seed=11)
    q2 = q.clone()
    q2[:, :, -1, :] += 50.0
    a = adaptive_spectral_global_mix(q, v, causal=True, gate_strength=0.5)
    b = adaptive_spectral_global_mix(q2, v, causal=True, gate_strength=0.5)
    assert (a - b)[:, :, :-1, :].abs().max() < 1e-9


def test_adaptive_hybrid_causal_does_not_see_the_future():
    """The leak must not survive through adaptive_hybrid_attention (#180)."""
    if _skip_if_no_torch():
        return
    seq, window = 32, 2
    q, k, v = _corr_sample(seq, seed=12)
    bumped = v.clone()
    bumped[:, :, -1, :] += 100.0
    base = adaptive_hybrid_attention(q, k, v, window=window, causal=True)
    moved = adaptive_hybrid_attention(q, k, bumped, window=window, causal=True)
    horizon = seq - 1 - window
    assert (moved - base)[:, :, :horizon, :].abs().max() < 1e-9


def test_adaptive_spectral_global_mix_noncausal_unchanged_by_new_flag():
    """Default branch must match pre-flag behavior."""
    if _skip_if_no_torch():
        return
    torch.manual_seed(13)
    q, k, v = _corr_sample(20, seed=13)
    a = adaptive_spectral_global_mix(q, v, freq_decay=0.5, gate_strength=0.2)
    b = adaptive_spectral_global_mix(
        q, v, freq_decay=0.5, gate_strength=0.2, causal=False
    )
    assert torch.allclose(a, b, atol=1e-12)


def test_correlation_spectral_global_mix_preserves_shape_and_finiteness():
    if _skip_if_no_torch():
        return
    q, k, v = _sample(seq=18, dim=6)
    out = correlation_spectral_global_mix(q, k, v, temperature=1.0, freq_decay=0.1)
    assert tuple(out.shape) == tuple(v.shape)
    assert torch.isfinite(out).all()


def test_correlation_spectral_global_mix_causal_does_not_see_the_future():
    """Regression for #154: the causal branch used a *circular* FFT convolution,
    so lag l > t wrapped onto v[seq + t - l] -- a future token. Perturbing only
    the last token of V must leave every earlier output untouched."""
    if _skip_if_no_torch():
        return
    seq = 16
    q, k, v = _corr_sample(seq, seed=0)
    bumped = v.clone()
    bumped[:, :, -1, :] += 100.0            # only the future-most token

    for freq_decay in (0.0, 0.7):
        base = correlation_spectral_global_mix(
            q, k, v, causal=True, freq_decay=freq_decay)
        moved = correlation_spectral_global_mix(
            q, k, bumped, causal=True, freq_decay=freq_decay)
        # every position before the last must be stable up to fp round-off
        assert (moved - base)[:, :, :-1, :].abs().max() < 1e-9, freq_decay
        # ...and the last position *must* react, or the kernel is degenerate
        assert (moved - base)[:, :, -1, :].abs().max() > 1e-6, freq_decay


def test_correlation_spectral_global_mix_causal_prefix_depends_only_on_prefix():
    """out[:cut] must be unchanged when the tail of V is replaced entirely."""
    if _skip_if_no_torch():
        return
    seq, cut = 16, 6
    q, k, v = _corr_sample(seq, seed=1)
    v2 = v.clone()
    v2[:, :, cut:, :] = torch.randn_like(v2[:, :, cut:, :])
    a = correlation_spectral_global_mix(q, k, v, causal=True)
    b = correlation_spectral_global_mix(q, k, v2, causal=True)
    assert (a - b)[:, :, :cut, :].abs().max() < 1e-9


def test_correlation_spectral_global_mix_causal_preserves_constant_v():
    """Each causal output is a convex combination of the visible past, so a
    constant V must come back unchanged (no shrink from the zero padding)."""
    if _skip_if_no_torch():
        return
    seq = 12
    q, k, _ = _corr_sample(seq, seed=2)
    v = torch.full((1, 2, seq, 4), 3.5, dtype=torch.float64)
    out = correlation_spectral_global_mix(q, k, v, causal=True)
    assert (out - 3.5).abs().max() < 1e-9


def test_correlation_spectral_global_mix_noncausal_still_wraps():
    """The non-causal branch is an intentional circular mixer -- unchanged."""
    if _skip_if_no_torch():
        return
    q, k, v = _corr_sample(16, seed=3)
    base = correlation_spectral_global_mix(q, k, v, causal=False)
    bumped = v.clone()
    bumped[:, :, -1, :] += 100.0
    moved = correlation_spectral_global_mix(q, k, bumped, causal=False)
    # wraparound means early positions DO move when causal=False
    assert (moved - base)[:, :, :-1, :].abs().max() > 1e-3


def test_correlation_hybrid_causal_does_not_see_the_future():
    """The leak must not survive through the hybrid wrapper (#154)."""
    if _skip_if_no_torch():
        return
    seq, window = 16, 2
    q, k, v = _corr_sample(seq, seed=4)
    bumped = v.clone()
    bumped[:, :, -1, :] += 100.0
    base = correlation_hybrid_attention(q, k, v, window=window, causal=True)
    moved = correlation_hybrid_attention(q, k, bumped, window=window, causal=True)
    # The local branch reads v[-1] only for queries within `window` of the end;
    # before that horizon nothing may move once the global branch is causal.
    horizon = seq - 1 - window
    assert (moved - base)[:, :, :horizon, :].abs().max() < 1e-9


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
