"""CPU-only tests for subspace._row_block static VRAM reservation.

Run:  python tests/test_row_block_static_budget.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace


class _FakeBackend:
    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free


def test_row_block_without_static_unchanged():
    backend = _FakeBackend(1024 * 1024)
    n, cols, item = 4096, 4096, 4
    frac = 0.5
    blk = subspace._row_block(n, cols, backend, item, frac, static_bytes=0)
    budget = int(backend.free_compute_bytes() * frac)
    assert blk == min(n, max(1, budget // (cols * item)))


def test_row_block_reserves_static_bytes():
    backend = _FakeBackend(1024 * 1024)
    n, cols, item = 4096, 4096, 4
    frac = 0.5
    static = 200 * 1024
    blk_with = subspace._row_block(n, cols, backend, item, frac, static_bytes=static)
    blk_without = subspace._row_block(n, cols, backend, item, frac, static_bytes=0)
    assert blk_with <= blk_without


def test_reconstruct_static_budget_formula():
    n, m = 32, 8
    item = np.dtype(np.float32).itemsize
    static = (n * m + m * m) * item
    backend = _FakeBackend(10**9)
    blk = subspace._row_block(n, n, backend, item, 0.3, static_bytes=static)
    budget = max(0, int(backend.free_compute_bytes() * 0.3) - static)
    assert blk == min(n, max(1, budget // (n * item)))


def test_compress_static_budget_formula():
    n, m = 24, 6
    item = np.dtype(np.float32).itemsize
    static = (n * m + m * m) * item
    backend = _FakeBackend(10**9)
    blk = subspace._row_block(n, n, backend, item, 0.3, static_bytes=static)
    budget = max(0, int(backend.free_compute_bytes() * 0.3) - static)
    assert blk == min(n, max(1, budget // (n * item)))


def test_row_block_raises_when_static_exhausts_budget():
    backend = _FakeBackend(1024)
    try:
        subspace._row_block(1024, 1024, backend, 4, 0.5, static_bytes=900)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


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
