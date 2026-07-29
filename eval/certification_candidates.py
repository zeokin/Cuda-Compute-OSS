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


_OUTPUT_CACHE = {}


def caches_first_output(q, k, v, *, causal: bool = False):
    """Invalid shortcut used to certify fresh-input timed validation."""
    key = (tuple(q.shape), tuple(k.shape), tuple(v.shape), bool(causal))
    if key not in _OUTPUT_CACHE:
        _OUTPUT_CACHE[key] = _production(q, k, v, causal=causal)
    return _OUTPUT_CACHE[key]


def mutates_runtime(q, k, v, *, causal: bool = False):
    """Invalid process mutation used to certify the runtime-state guard."""
    import torch

    torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark
    return _production(q, k, v, causal=causal)


__all__ = [
    "zeros",
    "nonfinite",
    "leaks_future",
    "redundant_work",
    "caches_first_output",
    "mutates_runtime",
]
