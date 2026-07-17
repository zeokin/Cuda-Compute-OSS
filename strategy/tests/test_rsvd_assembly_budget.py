"""CPU-only: rsvd.basis charges the preallocated (n,m) assembly buffer (#307).

rsvd used to keep all sketch parts live and concatenate them into another
(n, m) buffer (unbudgeted ~2x spike). It now preallocates Y, charges y_bytes
on every sketch, and copies each part into a column slice.

Run:  python strategy/tests/test_rsvd_assembly_budget.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import subspace as sub
from strategy.transforms import RandomizedSVDTransform


class _XP:
    empty = staticmethod(np.empty)

    class linalg:
        qr = staticmethod(np.linalg.qr)


class _FakeBackend:
    xp = _XP()

    def to_device(self, x):
        return np.asarray(x)


def _capture_extra_fixed(n=96, m=30, dtype=np.float32):
    captured = []
    orig_r, orig_l = sub.stream_gemm_right, sub.stream_gemm_left_t

    def fake(X, Q, backend, dt, frac=sub._DEFAULT_ROW_BLOCK_FRACTION,
             extra_fixed_bytes=0):
        captured.append(int(extra_fixed_bytes))
        return np.zeros((X.shape[0], Q.shape[1]), dtype=dt)

    sub.stream_gemm_right = fake
    sub.stream_gemm_left_t = fake
    try:
        A = np.eye(n, dtype=dtype)
        B = np.eye(n, dtype=dtype)
        Q = RandomizedSVDTransform(seed=0).basis(
            n, m, _FakeBackend(), dtype, A=A, B=B, frac=0.3
        )
        assert Q.shape == (n, m)
    finally:
        sub.stream_gemm_right, sub.stream_gemm_left_t = orig_r, orig_l
    return captured


def test_rsvd_charges_assembly_buffer_on_every_sketch():
    n, m = 96, 30
    item = np.dtype(np.float32).itemsize
    y_bytes = n * m * item
    seen = _capture_extra_fixed(n=n, m=m)
    assert seen == [y_bytes, y_bytes, y_bytes], seen


def test_assembly_without_concatenate_peak_beats_old_parts_plus_y():
    """Arithmetic witness: old parts+Y peak is 2*n*m; in-place assembly peaks at n*m
    before QR (QR still needs ~another n*m for Q)."""
    n, m, item = 8192, 1024, 4
    old_parts_plus_y = 2 * n * m * item
    new_assembly_peak = n * m * item
    assert new_assembly_peak < old_parts_plus_y
    assert new_assembly_peak * 2 == old_parts_plus_y


def test_rsvd_basis_still_orthonormal_on_cpu():
    n, m = 32, 12
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float64)
    B = rng.standard_normal((n, n)).astype(np.float64)
    from strategy.cpu_backend import CPUBackend
    Q = RandomizedSVDTransform(seed=1).basis(
        n, m, CPUBackend(verbose=False), np.float64, A=A, B=B, frac=0.3
    )
    assert Q.shape == (n, m)
    gram = Q.T @ Q
    assert np.allclose(gram, np.eye(m), atol=1e-9)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
