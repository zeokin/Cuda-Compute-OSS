"""subspace_matmul() must reject an out-of-range rank_m before building a Backend.

Every other argument the public API takes -- A/B shape, A/B dtype, config.dtype
agreement, and out's shape/dtype/writability -- is checked before any device
work. rank_m was the exception: Config can only check that it is an integer
(its valid range depends on n, which Config never sees), and the real
``1 <= m <= n`` guard lives in multiply_subspace(), which runs *after*
Backend(...). So an out-of-range rank_m demanded a GPU just to report itself.

A stub Backend that raises on construction pins the ordering, which is what
lets these run on CPU with no GPU/torch.
"""
from __future__ import annotations

import numpy as np
import pytest

import strategy
from strategy import Config, subspace

N = 8


class _ExplodingBackend:
    def __init__(self, *args, **kwargs):
        raise AssertionError("rank_m must be validated before Backend construction")


class _UnusedBackend:
    """multiply_subspace validates before it touches the backend, so a dummy is
    enough -- and any attribute access would fail loudly if that stopped being
    true."""


def _operands():
    return np.eye(N, dtype=np.float32), np.eye(N, dtype=np.float32)


@pytest.mark.parametrize("rank_m", (None, 1, N // 2, N), ids=repr)
def test_validator_accepts_valid_rank_m(rank_m):
    # None means "use default_rank_m(n)", so it is always valid.
    subspace.validate_rank_m(rank_m, N)


@pytest.mark.parametrize("rank_m", (0, -3, N + 1, 99), ids=repr)
def test_validator_rejects_out_of_range_rank_m(rank_m):
    with pytest.raises(RuntimeError, match=r"rank_m must be in \[1, n\]"):
        subspace.validate_rank_m(rank_m, N)


def test_both_entry_points_share_one_rule(monkeypatch):
    """The bound lives in exactly one validator. Both the public API and the
    core multiply must route through it and report it identically -- otherwise
    the two copies drift, which is what this refactor removes."""
    A, B = _operands()
    C = np.empty_like(A)
    cfg = Config(rank_m=N + 1, verbose=False)

    monkeypatch.setattr(strategy, "Backend", _ExplodingBackend)
    with pytest.raises(RuntimeError) as public_api:
        strategy.subspace_matmul(A, B, config=cfg)
    with pytest.raises(RuntimeError) as core:
        subspace.multiply_subspace(A, B, C, _UnusedBackend(), cfg)

    assert str(public_api.value) == str(core.value)


def test_public_api_delegates_to_the_shared_validator(monkeypatch):
    """Pin the delegation itself: if subspace_matmul ever grows its own copy of
    the bound, this fails even though the message still matches."""
    seen = []
    monkeypatch.setattr(subspace, "validate_rank_m",
                        lambda rank_m, n: seen.append((rank_m, n)))
    monkeypatch.setattr(strategy, "Backend", _ExplodingBackend)
    A, B = _operands()
    with pytest.raises(AssertionError, match="before Backend construction"):
        strategy.subspace_matmul(A, B, config=Config(rank_m=4, verbose=False))
    assert seen == [(4, N)]


@pytest.mark.parametrize("rank_m", (0, -3, N + 1, 99), ids=repr)
def test_rejects_out_of_range_rank_m_before_backend(monkeypatch, rank_m):
    monkeypatch.setattr(strategy, "Backend", _ExplodingBackend)
    A, B = _operands()
    with pytest.raises(RuntimeError, match=r"rank_m must be in \[1, n\]"):
        strategy.subspace_matmul(A, B, config=Config(rank_m=rank_m, verbose=False))


@pytest.mark.parametrize("rank_m", (1, N), ids=repr)
def test_valid_rank_m_still_reaches_backend(monkeypatch, rank_m):
    """1 and n are the valid extremes: they must pass the check and get through
    to Backend (our stub), guarding the bounds against an off-by-one."""
    monkeypatch.setattr(strategy, "Backend", _ExplodingBackend)
    A, B = _operands()
    with pytest.raises(AssertionError, match="before Backend construction"):
        strategy.subspace_matmul(A, B, config=Config(rank_m=rank_m, verbose=False))


def test_default_rank_m_is_unaffected(monkeypatch):
    """rank_m=None (the default M = min(n, max(64, n//8))) must not be rejected."""
    monkeypatch.setattr(strategy, "Backend", _ExplodingBackend)
    A, B = _operands()
    with pytest.raises(AssertionError, match="before Backend construction"):
        strategy.subspace_matmul(A, B, config=Config(verbose=False))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
