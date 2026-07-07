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


def spectral_global_mix(v, *, freq_decay: float = 1.0):
    """Cheap global mixer via FFT over the sequence dimension.

    This first reference version uses a deterministic low-pass filter rather
    than a learned kernel. It is intentionally simple: the goal is to establish
    an honest baseline for the global branch before any training or fusion work.
    """
    torch = _torch()
    if freq_decay < 0:
        raise ValueError("freq_decay must be >= 0")

    seq = v.shape[-2]
    vf = torch.fft.rfft(v, dim=-2)
    freqs = torch.arange(vf.shape[-2], device=v.device, dtype=v.real.dtype)
    gain = 1.0 / (1.0 + freq_decay * freqs)
    mixed = vf * gain.view(1, 1, -1, 1)
    return torch.fft.irfft(mixed, n=seq, dim=-2)


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
    if local_weight < 0 or global_weight < 0:
        raise ValueError("weights must be >= 0")
    total = local_weight + global_weight
    if total <= 0:
        raise ValueError("at least one branch weight must be positive")

    local = local_window_attention(
        q, k, v, window=window, causal=causal, block_size=block_size
    )
    global_ = spectral_global_mix(v, freq_decay=freq_decay)
    lw = local_weight / total
    gw = global_weight / total
    return lw * local + gw * global_
