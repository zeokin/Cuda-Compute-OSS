"""Hybrid attention playground: local exact + spectral global."""
from __future__ import annotations

import math


def _torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The attention playground requires PyTorch. Install the GPU extra: "
            "uv sync --extra gpu"
        ) from exc
    return torch


def _position_mask(q0: int, q1: int, k0: int, k1: int, *, window: int, causal: bool, device):
    torch = _torch()
    q_pos = torch.arange(q0, q1, device=device)[:, None]
    k_pos = torch.arange(k0, k1, device=device)[None, :]
    allowed = (q_pos - k_pos).abs() <= window
    if causal:
        allowed &= k_pos <= q_pos
    return allowed


def local_window_attention(q, k, v, *, window: int, causal: bool = False, block_size: int | None = None):
    """Exact local-window attention computed blockwise.

    Each query attends only to keys within ``window`` positions. The
    implementation keeps the attention exact inside that local band while
    avoiding materializing a full n x n score matrix.
    """
    torch = _torch()
    if window < 0:
        raise ValueError("window must be >= 0")

    batch, heads, seq, dim = q.shape
    block = block_size or min(max(64, window or 1), seq)
    out = torch.empty_like(v)

    for q0 in range(0, seq, block):
        q1 = min(seq, q0 + block)
        k0 = max(0, q0 - window)
        k1 = min(seq, q1 + (0 if causal else window))

        q_blk = q[:, :, q0:q1, :]
        k_blk = k[:, :, k0:k1, :]
        v_blk = v[:, :, k0:k1, :]

        scores = torch.matmul(q_blk, k_blk.transpose(-1, -2)) / math.sqrt(float(dim))
        mask = _position_mask(q0, q1, k0, k1, window=window, causal=causal, device=q.device)
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out[:, :, q0:q1, :] = torch.matmul(weights, v_blk)

    return out


def spectral_global_mix(v, *, freq_decay: float = 1.0, causal: bool = False):
    """Cheap global mixer via FFT over the sequence dimension.

    This first reference version uses a deterministic low-pass filter rather
    than a learned kernel. It is intentionally simple: the goal is to establish
    an honest baseline for the global branch before any training or fusion work.

    The default (non-causal) branch is a symmetric zero-phase low-pass, so each
    output depends on the whole sequence -- future included. With ``causal=True``
    it uses the causal counterpart of that first-order low-pass instead (see
    below), so ``out[t]`` depends only on ``v[0..t]``.
    """
    torch = _torch()
    if freq_decay < 0:
        raise ValueError("freq_decay must be >= 0")

    seq = v.shape[-2]
    # torch.fft does not support half precision, and fp16 is the default dtype
    # (AttentionSpec) -- upcast to a real float before the FFT and restore v's
    # dtype at the end, mirroring adaptive/correlation_spectral_global_mix.
    real_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
    v_work = v.to(real_dtype)

    if not causal:
        vf = torch.fft.rfft(v_work, dim=-2)
        freqs = torch.arange(vf.shape[-2], device=v.device, dtype=real_dtype)
        gain = 1.0 / (1.0 + freq_decay * freqs)
        mixed = vf * gain.view(1, 1, -1, 1)
        return torch.fft.irfft(mixed, n=seq, dim=-2).to(v.dtype)

    # Causal low-pass: the frequency gain 1/(1+freq_decay*f) is a first-order
    # low-pass; its causal counterpart is an exponential moving average with pole
    # a = 1/(1+freq_decay) (a=1 -> identity at freq_decay=0, matching the
    # non-causal branch's gain==1). Apply the geometric kernel g[l]=(1-a)^l as a
    # LINEAR (zero-padded) convolution so no future token leaks in, renormalized
    # by the partial kernel mass so early positions and constants are preserved
    # -- the same recipe as correlation_spectral_global_mix's causal branch.
    alpha = 1.0 / (1.0 + freq_decay)
    lags = torch.arange(seq, device=v.device, dtype=real_dtype)
    kernel = (1.0 - alpha) ** lags
    length = 2 * seq
    kf = torch.fft.rfft(kernel, n=length)
    vf = torch.fft.rfft(v_work, n=length, dim=-2)
    conv = torch.fft.irfft(vf * kf.view(1, 1, -1, 1), n=length, dim=-2)[..., :seq, :]
    mass = torch.cumsum(kernel, dim=0).clamp_min(1e-12).view(1, 1, -1, 1)
    return (conv / mass).to(v.dtype)


def adaptive_spectral_global_mix(
    q,
    v,
    *,
    freq_decay: float = 1.0,
    gate_strength: float = 0.25,
    causal: bool = False,
):
    """FFT global mixer with a deterministic input-adaptive frequency gate.

    The current fixed spectral branch applies the same low-pass filter to every
    input. This branch keeps the same FFT/IFFT structure but lets a summary of
    Q modulate the frequency response per batch, head, and channel.

    With ``causal=True``, the Q summary is a running mean (no future queries)
    and the V branch uses the same causal low-pass as ``spectral_global_mix``,
    so ``out[t]`` depends only on ``q[0..t]`` and ``v[0..t]``.
    """
    torch = _torch()
    if freq_decay < 0:
        raise ValueError("freq_decay must be >= 0")
    if gate_strength < 0:
        raise ValueError("gate_strength must be >= 0")

    seq = v.shape[-2]
    real_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
    v_work = v.to(real_dtype)
    q_work = q.to(real_dtype)

    if not causal:
        vf = torch.fft.rfft(v_work, dim=-2)
        freqs = torch.arange(vf.shape[-2], device=v.device, dtype=real_dtype)
        base_gain = 1.0 / (1.0 + freq_decay * freqs)

        q_summary = torch.tanh(q_work.mean(dim=-2, keepdim=True))
        gate = base_gain.view(1, 1, -1, 1) * (1.0 + gate_strength * q_summary)
        mixed = vf * gate.to(vf.dtype)
        out = torch.fft.irfft(mixed, n=seq, dim=-2)
        return out.to(v.dtype)

    q_cumsum = q_work.cumsum(dim=-2)
    counts = torch.arange(1, seq + 1, device=v.device, dtype=real_dtype).view(
        1, 1, -1, 1
    )
    q_summary = torch.tanh(q_cumsum / counts)

    alpha = 1.0 / (1.0 + freq_decay)
    lags = torch.arange(seq, device=v.device, dtype=real_dtype)
    kernel = (1.0 - alpha) ** lags
    length = 2 * seq
    kf = torch.fft.rfft(kernel, n=length)
    vf = torch.fft.rfft(v_work, n=length, dim=-2)
    conv = torch.fft.irfft(vf * kf.view(1, 1, -1, 1), n=length, dim=-2)[..., :seq, :]
    mass = torch.cumsum(kernel, dim=0).clamp_min(1e-12).view(1, 1, -1, 1)
    base = conv / mass
    gate = 1.0 + gate_strength * q_summary
    return (base * gate).to(v.dtype)


def correlation_spectral_global_mix(
    q,
    k,
    v,
    *,
    temperature: float = 1.0,
    freq_decay: float = 0.0,
    causal: bool = False,
):
    """Global FFT mixer using a Q/K-derived lag kernel.

    The kernel is built from average Q/K cross-correlation over sequence lags,
    then applied to V as a circular convolution in frequency space.
    """
    torch = _torch()
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if freq_decay < 0:
        raise ValueError("freq_decay must be >= 0")

    seq = v.shape[-2]
    if causal:
        # #154 made the V-convolution causal, but the lag kernel is still built
        # from whole-sequence Q/K statistics (mean over all positions, a full-
        # sequence FFT cross-correlation, softmax over all lags), so this single
        # shared kernel still depends on future q/k -- out[t] leaks q/k[t+1..].
        # A shared kernel cannot be causal, so use the causal spectral low-pass
        # (mirrors the adaptive -> spectral and topk -> pooled causal fallbacks).
        return spectral_global_mix(v, freq_decay=freq_decay, causal=True)

    real_dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
    q_work = (q - q.mean(dim=-2, keepdim=True)).to(real_dtype)
    k_work = (k - k.mean(dim=-2, keepdim=True)).to(real_dtype)
    v_work = v.to(real_dtype)

    q_norm = torch.linalg.vector_norm(q_work, dim=-1, keepdim=True).clamp_min(1e-6)
    k_norm = torch.linalg.vector_norm(k_work, dim=-1, keepdim=True).clamp_min(1e-6)
    q_work = q_work / q_norm
    k_work = k_work / k_norm

    qf = torch.fft.fft(q_work, dim=-2)
    kf = torch.fft.fft(k_work, dim=-2)
    corr = torch.fft.ifft(qf * torch.conj(kf), dim=-2).real.mean(dim=-1).abs()

    if causal:
        lag_idx = torch.arange(seq, device=v.device)
        corr = corr.masked_fill(lag_idx.view(1, 1, -1) > seq // 2, float("-inf"))

    kernel = torch.softmax(corr / temperature, dim=-1)
    delta = torch.zeros_like(kernel)
    delta[..., 0] = 1.0
    kernel = 0.5 * delta + 0.5 * kernel
    if freq_decay > 0:
        freqs = torch.arange(seq, device=v.device, dtype=real_dtype)
        decay = 1.0 / (1.0 + freq_decay * torch.minimum(freqs, seq - freqs))
        kernel = kernel * decay.view(1, 1, -1)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    if causal:
        # A circular convolution computes out[t] = sum_l kernel[l] * v[(t-l) % seq],
        # so every lag l > t wraps onto v[seq + t - l] -- a FUTURE token. Masking
        # far lags above does not prevent that. Zero-pad to 2*seq so the FFT
        # computes a LINEAR convolution and only lags l <= t contribute.
        length = 2 * seq
        vf = torch.fft.fft(v_work, n=length, dim=-2)
        kernel_f = torch.fft.fft(kernel, n=length, dim=-1)[..., None]
        out = torch.fft.ifft(vf * kernel_f, dim=-2).real[..., :seq, :]
        # Only the first t+1 taps land on real tokens, so renormalize by the
        # kernel mass actually applied at t -- keeping each output a convex
        # combination of the visible past (cf. _pooled_landmarks' denom).
        mass = torch.cumsum(kernel, dim=-1).clamp_min(1e-12)[..., None]
        out = out / mass
    else:
        # Non-causal: wraparound is intentional (a global circular mixer).
        vf = torch.fft.fft(v_work, dim=-2)
        kernel_f = torch.fft.fft(kernel, dim=-1)[..., None]
        out = torch.fft.ifft(vf * kernel_f, dim=-2).real
    return out.to(v.dtype)


def _pad_to_blocks(x, *, blocks: int):
    torch = _torch()
    seq = x.shape[-2]
    block = math.ceil(seq / blocks)
    padded = block * blocks
    if padded == seq:
        return x, seq, block
    pad_shape = (*x.shape[:-2], padded - seq, x.shape[-1])
    pad = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-2), seq, block


def _pooled_landmarks(k, v, *, num_landmarks: int):
    torch = _torch()
    batch, heads, seq, dim = k.shape
    landmarks = min(num_landmarks, seq)
    k_pad, _seq, block = _pad_to_blocks(k, blocks=landmarks)
    v_pad, _seq, _block = _pad_to_blocks(v, blocks=landmarks)

    k_blocks = k_pad.reshape(batch, heads, landmarks, block, dim)
    v_blocks = v_pad.reshape(batch, heads, landmarks, block, dim)

    # Real (non-padding) token count per block: block i covers [i*block, (i+1)*block),
    # so it holds clamp(seq - i*block, 0, block) real tokens. This varies per block --
    # when the tail padding spans more than one block (block*landmarks - seq >= block),
    # the boundary block is partial and trailing blocks are pure padding, so we cannot
    # assume a single fixed `block` with only the LAST block corrected.
    idx = torch.arange(landmarks, device=k.device)
    counts = (seq - idx * block).clamp(min=0, max=block)      # (landmarks,) real tokens
    valid = counts > 0                                        # fully-padding blocks -> no landmark
    denom = counts.clamp_min(1).to(k.real.dtype).view(1, 1, landmarks, 1)

    k_landmarks = k_blocks.sum(dim=-2) / denom
    v_landmarks = v_blocks.sum(dim=-2) / denom
    # Causal position = index of the LAST real token in the block; invalid (all-padding)
    # blocks get `seq` (>= every query index) so the causal mask also drops them.
    positions = torch.where(valid, idx * block + counts - 1, torch.full_like(idx, seq))
    return k_landmarks, v_landmarks, positions, valid


def _topk_landmarks(q, k, v, *, num_landmarks: int):
    torch = _torch()
    batch, heads, seq, dim = k.shape
    landmarks = min(num_landmarks, seq)

    # Use alignment with the mean query direction as a cheap, deterministic
    # proxy for "globally relevant" tokens, then break ties with key magnitude.
    # This keeps selection query-aware without materializing full n x n scores.
    q_mean = q.to(torch.float32).mean(dim=-2, keepdim=True)
    align = torch.abs((k.to(torch.float32) * q_mean).sum(dim=-1))
    norm = torch.linalg.vector_norm(k.to(torch.float32), dim=-1)
    scores = align + 0.05 * norm
    topk = torch.topk(scores, k=landmarks, dim=-1).indices
    topk, _ = torch.sort(topk, dim=-1)

    gather = topk[..., None].expand(batch, heads, landmarks, dim)
    k_landmarks = torch.gather(k, dim=-2, index=gather)
    v_landmarks = torch.gather(v, dim=-2, index=gather)
    return k_landmarks, v_landmarks, topk


def landmark_global_attention(
    q,
    k,
    v,
    *,
    num_landmarks: int = 64,
    causal: bool = False,
    policy: str = "pooled",
):
    """Approximate global attention by attending to selected K/V landmarks."""
    torch = _torch()
    if num_landmarks <= 0:
        raise ValueError("num_landmarks must be > 0")
    if policy not in {"pooled", "topk"}:
        raise ValueError("policy must be one of: pooled, topk")
    if policy == "topk" and causal:
        # topk selects a single global landmark set from a mean over ALL queries
        # and a top-k over ALL keys, so out[t] would depend on future q/k -- the
        # later position mask hides landmark positions, not the future-tainted
        # selection. A global set cannot be causal, so fall back to the position-
        # based pooled selection, which is causally correct.
        policy = "pooled"

    _batch, _heads, seq, dim = q.shape
    valid = None
    if policy == "pooled":
        k_landmarks, v_landmarks, positions, valid = _pooled_landmarks(
            k, v, num_landmarks=num_landmarks
        )
    else:
        k_landmarks, v_landmarks, positions = _topk_landmarks(
            q, k, v, num_landmarks=num_landmarks
        )

    scores = torch.matmul(q, k_landmarks.transpose(-1, -2)) / math.sqrt(float(dim))
    if valid is not None:
        # Drop pooled landmarks that are entirely padding (no real tokens) -- an
        # all-zero landmark would otherwise still collect softmax weight and drag
        # the output toward zero. Applies in both causal and non-causal modes.
        scores = scores.masked_fill(~valid.view(1, 1, 1, -1), float("-inf"))
    if causal:
        q_pos = torch.arange(seq, device=q.device)[:, None]
        if policy == "pooled":
            landmark_pos = positions.view(1, 1, -1)
        else:
            landmark_pos = positions
        mask = landmark_pos[:, :, None, :] > q_pos[None, None, :, :]
        scores = scores.masked_fill(mask, float("-inf"))
        # An early query can precede every landmark, masking its whole row to
        # -inf; softmax over an all -inf row is NaN and would poison the output
        # and the benchmark's quality metrics. Give those rows zero weight (no
        # visible landmark -> no global contribution) instead.
        all_masked = mask.all(dim=-1, keepdim=True)
        weights = torch.softmax(scores.masked_fill(all_masked, 0.0), dim=-1)
        weights = weights.masked_fill(all_masked, 0.0)
    else:
        weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v_landmarks)


def _normalized_weights(local_weight: float, global_weight: float) -> tuple[float, float]:
    if local_weight < 0 or global_weight < 0:
        raise ValueError("weights must be >= 0")
    total = local_weight + global_weight
    if total <= 0:
        raise ValueError("at least one branch weight must be positive")
    return local_weight / total, global_weight / total


def hybrid_attention(
    q,
    k,
    v,
    *,
    window: int,
    causal: bool = False,
    block_size: int | None = None,
    local_weight: float = 0.85,
    global_weight: float = 0.15,
    freq_decay: float = 1.0,
):
    """Combine exact local attention with cheap spectral global mixing."""
    lw, gw = _normalized_weights(local_weight, global_weight)
    local = local_window_attention(
        q, k, v, window=window, causal=causal, block_size=block_size
    )
    if gw == 0:
        return local
    global_ = spectral_global_mix(v, freq_decay=freq_decay, causal=causal)
    return lw * local + gw * global_


def adaptive_hybrid_attention(
    q,
    k,
    v,
    *,
    window: int,
    causal: bool = False,
    block_size: int | None = None,
    local_weight: float = 0.85,
    global_weight: float = 0.15,
    freq_decay: float = 1.0,
    gate_strength: float = 0.25,
):
    """Combine local exact attention with an adaptive spectral global branch."""
    lw, gw = _normalized_weights(local_weight, global_weight)
    local = local_window_attention(
        q, k, v, window=window, causal=causal, block_size=block_size
    )
    if gw == 0:
        return local
    global_ = adaptive_spectral_global_mix(
        q, v, freq_decay=freq_decay, gate_strength=gate_strength, causal=causal
    )
    return lw * local + gw * global_


def correlation_hybrid_attention(
    q,
    k,
    v,
    *,
    window: int,
    causal: bool = False,
    block_size: int | None = None,
    local_weight: float = 0.85,
    global_weight: float = 0.15,
    temperature: float = 1.0,
    freq_decay: float = 0.0,
):
    """Combine local exact attention with a Q/K correlation FFT global branch."""
    lw, gw = _normalized_weights(local_weight, global_weight)
    local = local_window_attention(
        q, k, v, window=window, causal=causal, block_size=block_size
    )
    if gw == 0:
        return local
    global_ = correlation_spectral_global_mix(
        q, k, v, temperature=temperature, freq_decay=freq_decay, causal=causal
    )
    return lw * local + gw * global_


def landmark_hybrid_attention(
    q,
    k,
    v,
    *,
    window: int,
    causal: bool = False,
    block_size: int | None = None,
    local_weight: float = 0.85,
    global_weight: float = 0.15,
    num_landmarks: int = 64,
    landmark_policy: str = "pooled",
):
    """Combine local exact attention with landmark-based global attention."""
    lw, gw = _normalized_weights(local_weight, global_weight)
    local = local_window_attention(
        q, k, v, window=window, causal=causal, block_size=block_size
    )
    if gw == 0:
        return local
    global_ = landmark_global_attention(
        q, k, v, num_landmarks=num_landmarks, causal=causal, policy=landmark_policy
    )
    return lw * local + gw * global_
