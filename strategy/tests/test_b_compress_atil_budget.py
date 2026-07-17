"""CPU-only: compress(B) must charge live Atil (#304).

multiply_subspace keeps Atil = compress(A) on device while compressing B.
compress() already budgets Q + acc + product; without extra_fixed_bytes for
Atil, B's row-block under-counts by m*m on static free-memory backends (MPS).

Run:  python strategy/tests/test_b_compress_atil_budget.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import subspace
from strategy.cpu_backend import CPUBackend
from strategy.config import Config

FRAC = subspace._DEFAULT_ROW_BLOCK_FRACTION


def _spy_fixed_bytes(call):
    seen = {}
    real = subspace._row_block

    def spy(bn, cols, backend, item_bytes, frac=FRAC, out_cols=0, fixed_bytes=0,
            **kw):
        seen["fixed_bytes"] = fixed_bytes
        return real(bn, cols, backend, item_bytes, frac,
                    out_cols=out_cols, fixed_bytes=fixed_bytes, **kw)

    subspace._row_block = spy
    try:
        call()
    finally:
        subspace._row_block = real
    return seen["fixed_bytes"]


def test_compress_extra_fixed_bytes_is_added():
    n, m, dtype = 64, 16, np.float32
    item = np.dtype(dtype).itemsize
    X = np.zeros((n, n), dtype=dtype)
    Q = np.zeros((n, m), dtype=dtype)
    extra = m * m * item
    base = _spy_fixed_bytes(
        lambda: subspace.compress(X, Q, CPUBackend(verbose=False), dtype)
    )
    with_extra = _spy_fixed_bytes(
        lambda: subspace.compress(
            X, Q, CPUBackend(verbose=False), dtype, extra_fixed_bytes=extra
        )
    )
    assert base == (n * m + 2 * m * m) * item
    assert with_extra == base + extra


def test_multiply_subspace_charges_atil_on_second_compress(monkeypatch):
    """B-compress must see extra_fixed_bytes == sizeof(Atil)."""
    n, m = 32, 8
    charges = []
    real = subspace.compress

    def capture(X, Q, backend, dtype, frac=FRAC, extra_fixed_bytes=0):
        charges.append(int(extra_fixed_bytes))
        return real(X, Q, backend, dtype, frac, extra_fixed_bytes=extra_fixed_bytes)

    class FakeTransform:
        name = "fake"

        def basis(self, n_, m_, backend, dtype, A=None, B=None, frac=None):
            return np.eye(n_, m_, dtype=dtype)

        def basis_flops(self, n_, m_):
            return 0.0

    monkeypatch.setattr(subspace, "compress", capture)
    monkeypatch.setattr(subspace, "get_transform", lambda *a, **k: FakeTransform())
    monkeypatch.setattr(
        subspace, "reconstruct",
        lambda *a, **k: None,
    )

    A = np.eye(n, dtype=np.float64)
    B = np.eye(n, dtype=np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    cfg = Config(rank_m=m, dtype="fp64", vram_fraction=FRAC, verbose=False)
    subspace.multiply_subspace(A, B, C, CPUBackend(verbose=False), cfg)

    assert charges == [0, m * m * 8], charges


def test_atil_charge_shrinks_b_block_on_static_free():
    """Witness: charging Atil reduces the chosen block vs the uncharged path."""
    class Static:
        def __init__(self, f):
            self._f = f

        def free_compute_bytes(self):
            return self._f

    n, m, item = 4096, 2048, 4
    free = 300 * 1024**2
    base = (n * m + 2 * m * m) * item
    atil = m * m * item
    without = subspace._row_block(
        n, n, Static(free), item, FRAC, out_cols=m, fixed_bytes=base
    )
    with_atil = subspace._row_block(
        n, n, Static(free), item, FRAC, out_cols=m, fixed_bytes=base + atil
    )
    assert with_atil < without
    # Live peak with Atil must not exceed the fraction budget when using with_atil.
    per_row = (n + m) * item  # staged cols + out_cols (approx lower bound)
    budget = int(free * FRAC)
    assert with_atil * per_row + base + atil <= budget


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
