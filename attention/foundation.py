"""Exact attention implementation optimized by the foundation phase.

This module is intentionally small.  The protected evaluator imports
``attention_forward`` from both ``main`` and a candidate PR checkout and
compares them on the frozen attention workload.  Contributor PRs may optimize
this implementation, but they may not modify the evaluator or its manifest.
"""
from __future__ import annotations


def attention_forward(q, k, v, *, causal: bool = False):
    """Compute exact scaled-dot-product attention without dropout.

    ``q`` may have a different sequence length from ``k``/``v``.  Decode
    workloads pass only the already-visible KV cache and therefore use
    ``causal=False`` for a one-token query.
    """
    try:
        import torch.nn.functional as functional
    except Exception as exc:  # pragma: no cover - exercised without torch
        raise RuntimeError(
            "The attention foundation requires PyTorch. Install the GPU extra."
        ) from exc
    return functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=causal,
    )


__all__ = ["attention_forward"]
