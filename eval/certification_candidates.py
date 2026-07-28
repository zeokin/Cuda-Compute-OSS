"""Deliberately bad candidates used only to certify protected GPU gates."""
from __future__ import annotations


def _production(q, k, v, *, causal: bool):
    import torch.nn.functional as functional
    return functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=causal
    )


def zeros(q, k, v, *, causal: bool = False):
    return _production(q, k, v, causal=causal).zero_()


def nonfinite(q, k, v, *, causal: bool = False):
    output = _production(q, k, v, causal=causal)
    return output.fill_(float("nan"))


def leaks_future(q, k, v, *, causal: bool = False):
    return _production(q, k, v, causal=False)


def redundant_work(q, k, v, *, causal: bool = False):
    _production(q, k, v, causal=causal)
    return _production(q, k, v, causal=causal)


__all__ = ["zeros", "nonfinite", "leaks_future", "redundant_work"]
