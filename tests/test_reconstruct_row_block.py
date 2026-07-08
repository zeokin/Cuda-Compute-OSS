"""CPU-only test for subspace.reconstruct()'s row-block VRAM budgeting.

reconstruct() streams its (n x n) output row-by-row; the row-block must be sized
from the dtype the loop tensors actually live in (compute_dtype), not the output
dtype. For fp16 inputs compute_dtype is bumped to fp32, so sizing from out_dtype
(#66) under-budgets the block 2x. This is pure arithmetic — no GPU needed.

Run:  python tests/test_reconstruct_row_block.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace


class _FakeBackend:
    """Minimal CPU backend: numpy matmul + a fixed free-memory figure."""

    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free

    def matmul(self, a, b):
        return np.asarray(a) @ np.asarray(b)

    def to_host(self, x):
        return np.asarray(x)


def _capture_row_block_item_bytes(run):
    """Run `run()` with subspace._row_block spied so we can read the item_bytes
    it was sized with, then restore the original."""
    captured = []
    real = subspace._row_block

    def spy(n, cols, backend, item_bytes, frac=subspace._DEFAULT_ROW_BLOCK_FRACTION):
        captured.append(item_bytes)
        return real(n, cols, backend, item_bytes, frac)

    subspace._row_block = spy
    try:
        run()
    finally:
        subspace._row_block = real
    return captured


def _reconstruct(out_dtype, compute_dtype):
    n, m = 16, 4
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0].astype(np.float32)
    Ctil = rng.standard_normal((m, m)).astype(np.float32)
    C = np.zeros((n, n), dtype=out_dtype)
    subspace.reconstruct(Ctil, Q, C, _FakeBackend(10**9), out_dtype, compute_dtype=compute_dtype)


def test_reconstruct_sizes_row_block_from_compute_dtype():
    # Regression for #66: fp16 output but fp32 compute must size the row-block
    # from fp32 (4 bytes). Pre-fix it used out_dtype's 2 bytes -> 2x under-budget.
    captured = _capture_row_block_item_bytes(lambda: _reconstruct(np.float16, np.float32))
    assert captured, "reconstruct did not call _row_block"
    assert captured[-1] == np.dtype(np.float32).itemsize  # 4, not 2


def test_reconstruct_item_bytes_defaults_to_out_dtype():
    # compute_dtype=None keeps the pre-existing behavior (out_dtype) for callers
    # that don't pass it — the fp64 path where compute == output.
    captured = _capture_row_block_item_bytes(lambda: _reconstruct(np.float64, None))
    assert captured
    assert captured[-1] == np.dtype(np.float64).itemsize  # 8


def test_reconstruct_output_is_correct_on_cpu():
    # The dtype-sizing change must not alter the math: Q @ Ctil @ Q^T.
    n, m = 24, 6
    rng = np.random.default_rng(1)
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0].astype(np.float64)
    Ctil = rng.standard_normal((m, m)).astype(np.float64)
    C = np.zeros((n, n), dtype=np.float64)
    subspace.reconstruct(Ctil, Q, C, _FakeBackend(10**9), np.float64, compute_dtype=np.float64)
    expected = Q @ Ctil @ Q.T
    assert float(np.linalg.norm(C - expected) / (np.linalg.norm(expected) or 1.0)) < 1e-12


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
