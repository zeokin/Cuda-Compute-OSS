"""CPU-only tests: the streaming primitives charge their resident Q operand.

reconstruct already charges its resident Q (n, m) and Ctil (m, m) up front
(#216), because on MPS free_compute_bytes() is a static ceiling that never
reflects a caller-allocated resident buffer. stream_gemm_right, stream_gemm_left_t
and compress take the same resident Q operand but did not charge it -- so on MPS
their block was under-budgeted by n*m. These tests assert each now folds the
Q operand (n*m) into its fixed_bytes.

Run:  python tests/test_streaming_resident_q_budget.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace
from strategy.cpu_backend import CPUBackend

FRAC = subspace._DEFAULT_ROW_BLOCK_FRACTION


def _fixed_bytes_for(primitive):
    seen = {}
    real = subspace._row_block

    def spy(bn, cols, backend, item_bytes, frac=FRAC, out_cols=0, fixed_bytes=0,
            transient_cols=0):
        seen["fixed_bytes"] = fixed_bytes
        return real(bn, cols, backend, item_bytes, frac, out_cols=out_cols,
                    fixed_bytes=fixed_bytes, transient_cols=transient_cols)

    subspace._row_block = spy
    try:
        primitive()
    finally:
        subspace._row_block = real
    return seen["fixed_bytes"]


def test_each_primitive_charges_the_resident_q_operand():
    n, m = 64, 16
    item = np.dtype(np.float64).itemsize
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, n))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    bk = CPUBackend(verbose=False)

    # stream_gemm_right: resident out (n,m) + resident Q (n,m) = 2*n*m
    fr = _fixed_bytes_for(lambda: subspace.stream_gemm_right(X, Q, bk, np.float64))
    assert fr == (n * m + n * m) * item, fr

    # stream_gemm_left_t: acc (n,m) + product (n,m) + resident Q (n,m) = 3*n*m
    fl = _fixed_bytes_for(lambda: subspace.stream_gemm_left_t(X, Q, bk, np.float64))
    assert fl == (2 * n * m + n * m) * item, fl

    # compress: acc (m,m) + product (m,m) + resident Q (n,m) = 2*m*m + n*m
    fc = _fixed_bytes_for(lambda: subspace.compress(X, Q, bk, np.float64))
    assert fc == (2 * m * m + n * m) * item, fc


def test_q_operand_makes_the_block_smaller():
    """Charging the resident Q must shrink the chosen block vs omitting it."""
    class Static:
        def __init__(self, f): self._f = f
        def free_compute_bytes(self): return self._f

    n, m, item = 8192, 1024, 4
    bk = Static(300 * 1024**2)
    without_q = subspace._row_block(n, n, bk, item, FRAC, out_cols=m,
                                    fixed_bytes=2 * m * m * item, transient_cols=n)
    with_q = subspace._row_block(n, n, bk, item, FRAC, out_cols=m,
                                 fixed_bytes=(2 * m * m + n * m) * item, transient_cols=n)
    assert with_q < without_q


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
