"""Exact attention reference operator."""
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


def exact_attention(q, k, v, *, causal: bool = False):
    """Return exact scaled dot-product attention.

    Shapes:
        q, k, v: (batch, heads, seq, dim)
        out:     (batch, heads, seq, dim)
    """
    torch = _torch()
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(d))
    if causal:
        seq = q.shape[-2]
        mask = torch.ones((seq, seq), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
