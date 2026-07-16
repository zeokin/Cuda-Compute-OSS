"""CPU-only tests for compress()'s device budget (issue #220).

compress accumulates Q^T X Q into a resident (m, m) buffer, and at the peak of
``acc += matmul(...)`` holds TWO (m, m) buffers -- acc and the product that
cannot alias it -- neither scaling with the block. The row-block budget must
therefore charge 2*m*m up front; charging only the product (m*m) leaves the
accumulator unbudgeted (and invisible on MPS, where free_compute_bytes() is a
static ceiling), so the block overshoots and can OOM. compress was the one
streaming primitive still missing this after #203/#205/#216.

These tests assert compress hands _row_block the full 2*m*m charge, that the
old m*m charge overshoots the real peak, and that the change leaves the math of
Q^T X Q intact. Pure NumPy; no GPU needed.

Run:  python tests/test_compress_budget.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace
from strategy.cpu_backend import CPUBackend

FRAC = subspace._DEFAULT_ROW_BLOCK_FRACTION


def _compress_fixed_bytes(n, m, dtype):
    """Spy on _row_block to capture the fixed_bytes compress charges for an
    (n, n) X and (n, m) Q, driving the real compress() on a CPU backend."""
    seen = {}
    real = subspace._row_block

    def spy(bn, cols, backend, item_bytes, frac=FRAC, out_cols=0, fixed_bytes=0,
            **kw):
        seen["fixed_bytes"] = fixed_bytes
        seen["item_bytes"] = item_bytes
        return real(bn, cols, backend, item_bytes, frac,
                    out_cols=out_cols, fixed_bytes=fixed_bytes, **kw)

    subspace._row_block = spy
    try:
        X = np.zeros((n, n), dtype=dtype)
        Q = np.zeros((n, m), dtype=dtype)
        subspace.compress(X, Q, CPUBackend(verbose=False), dtype)
    finally:
        subspace._row_block = real
    return seen


def test_compress_charges_accumulator_and_product():
    # Both the resident acc and the per-step product are (m, m): 2*m*m, not m*m.
    for n, m, dtype in [(64, 16, np.float32), (128, 32, np.float64), (256, 256, np.float32)]:
        seen = _compress_fixed_bytes(n, m, dtype)
        item = np.dtype(dtype).itemsize
        assert seen["fixed_bytes"] == 2 * m * m * item, (n, m, dtype, seen)
        assert seen["fixed_bytes"] != m * m * item      # explicitly not the old model


def test_old_single_mm_charge_overshoots_budget():
    """Witness: with only the product charged, the block plus BOTH resident (m, m)
    buffers exceeds the budget it was sized against (a static MPS-style free)."""
    class Static:
        def __init__(self, f): self._f = f
        def free_compute_bytes(self): return self._f

    n, m, item = 4096, 2048, 4
    free = 220 * 1024**2
    budget = int(free * FRAC)
    per_row = (n + m) * item
    resident_plus_product = 2 * m * m * item

    old_blk = subspace._row_block(n, n, Static(free), item, FRAC, out_cols=m, fixed_bytes=m * m * item)
    assert old_blk * per_row + resident_plus_product > budget        # overshoots

    new_blk = subspace._row_block(n, n, Static(free), item, FRAC, out_cols=m, fixed_bytes=2 * m * m * item)
    assert new_blk < old_blk
    assert new_blk * per_row + resident_plus_product <= budget       # now fits


def test_compress_math_unchanged_when_blocked():
    """Q^T X Q must be exact even when the (tight) budget forces multiple blocks."""
    class Tight(CPUBackend):
        def free_compute_bytes(self): return 120_000

    n, m = 96, 12
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, n))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    got = np.asarray(subspace.compress(X, Q, Tight(verbose=False), np.float64))
    expected = Q.T @ X @ Q
    rel = np.linalg.norm(got - expected) / (np.linalg.norm(expected) or 1.0)
    assert rel < 1e-12, rel


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
