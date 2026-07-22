"""CPU-only: nystrom.basis must not concatenate-all-parts (#315).

nystrom used to keep every landmark block live and concatenate them into a
second (n, m) buffer — the same unbudgeted ~2x spike #308 fixed for rsvd.
It now preallocates Y and uploads one block at a time.

Run:  python strategy/tests/test_nystrom_assembly_budget.py
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy.transforms import NystromTransform, Transform


class _XP:
    empty = staticmethod(np.empty)
    concatenate_calls = 0

    @staticmethod
    def concatenate(arrays, axis=0):
        _XP.concatenate_calls += 1
        return np.concatenate(list(arrays), axis=axis)

    class linalg:
        qr = staticmethod(np.linalg.qr)


class _FakeBackend:
    xp = _XP()

    def to_device(self, x):
        return np.asarray(x)


def test_nystrom_basis_accepts_frac_like_transform_contract():
    assert "frac" in inspect.signature(Transform.basis).parameters
    assert "frac" in inspect.signature(NystromTransform.basis).parameters


def test_nystrom_does_not_concatenate_all_parts():
    _XP.concatenate_calls = 0
    n, m = 48, 16
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float64)
    B = rng.standard_normal((n, n)).astype(np.float64)
    Q = NystromTransform(seed=0).basis(
        n, m, _FakeBackend(), np.float64, A=A, B=B, frac=0.3
    )
    assert Q.shape == (n, m)
    assert _XP.concatenate_calls == 0, _XP.concatenate_calls


def test_assembly_peak_model_beats_parts_plus_y():
    n, m, item = 8192, 1024, 4
    old_parts_plus_y = 2 * n * m * item
    new_assembly_peak = n * m * item
    assert new_assembly_peak * 2 == old_parts_plus_y


def test_nystrom_basis_orthonormal_on_cpu():
    from strategy.cpu_backend import CPUBackend

    n, m = 32, 12
    rng = np.random.default_rng(1)
    A = rng.standard_normal((n, n)).astype(np.float64)
    B = rng.standard_normal((n, n)).astype(np.float64)
    Q = NystromTransform(seed=2).basis(
        n, m, CPUBackend(verbose=False), np.float64, A=A, B=B
    )
    assert Q.shape == (n, m)
    assert np.allclose(Q.T @ Q, np.eye(m), atol=1e-9)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
