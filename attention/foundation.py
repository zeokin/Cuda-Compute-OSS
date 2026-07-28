"""Exact attention implementation optimized by the foundation phase.

This module is intentionally small.  The protected evaluator imports
``attention_forward`` from both ``main`` and a candidate PR checkout and
compares them on the frozen attention workload.  Contributor PRs may optimize
this implementation, but they may not modify the evaluator or its manifest.
"""
from __future__ import annotations


def _decode_backend_priority():
    """Backend priority for the one-token (``q_len == 1``) decode path.

    Returns an ordered ``SDPBackend`` list, or ``None`` when this torch build
    has no backend-selection context manager (fall back to the default policy).

    FlashAttention's kernels are tiled for prefill, where the query axis is long
    enough to fill a CTA; a ``q_len == 1`` decode leaves that tiling almost
    entirely idle (issue #343).  cuDNN's fused attention and the
    memory-efficient backend have decode-shaped paths that do not pay the
    prefill tiling overhead, so we ask for them first.  ``MATH`` is kept last so
    there is ALWAYS a backend that supports the shape -- the dispatch can never
    fail to find one, and whatever it picks computes the identical attention
    (all SDPA backends are exact to within the manifest's tolerance).
    """
    try:
        from torch.nn.attention import SDPBackend
    except Exception:  # pragma: no cover - very old torch without the enum
        return None
    return [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]


def attention_forward(q, k, v, *, causal: bool = False):
    """Compute exact scaled-dot-product attention without dropout.

    ``q`` may have a different sequence length from ``k``/``v``.  Decode
    workloads pass only the already-visible KV cache and therefore use
    ``causal=False`` for a one-token query.

    Prefill is left on PyTorch's default backend policy.  For a one-token decode
    (``q_len == 1``) we express a decode-first backend priority so the dispatcher
    does not spend FlashAttention's prefill-tuned tiling on a single query row
    (issue #343).  The selection is purely a performance hint: the math kernel is
    always in the priority list as a guaranteed fallback, and every SDPA backend
    returns the same attention within tolerance, so output is unchanged.
    """
    try:
        import torch.nn.functional as functional
    except Exception as exc:  # pragma: no cover - exercised without torch
        raise RuntimeError(
            "The attention foundation requires PyTorch. Install the GPU extra."
        ) from exc

    def _default():
        return functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=causal
        )

    # Only the one-token decode path is retuned; everything else keeps main's
    # exact behaviour and dispatch.
    if q.shape[-2] != 1:
        return _default()

    backends = _decode_backend_priority()
    if backends is None:
        return _default()

    try:
        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(backends, set_priority=True):
            return functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=causal
            )
    except Exception:  # noqa: BLE001 - any dispatch/availability issue -> default
        # A build without ``set_priority``, or a backend that rejects the shape,
        # must never change the result: fall back to the exact default path.
        return _default()


__all__ = ["attention_forward"]
