"""CPU-only test for subspace.reconstruct()'s row-block VRAM budgeting.

reconstruct() streams its (n x n) output row-by-row; the row-block must be sized
from the dtype the loop tensors actually live in (compute_dtype), not the output
dtype. For fp16 inputs compute_dtype is bumped to fp32, so sizing from out_dtype
(#66) under-budgets the block 2x. This is pure arithmetic — no GPU needed.

Q (n, m) and Ctil (m, m) are also both fully resident on the device for the
entire loop -- unlike the streamed inputs elsewhere in this module, they arrive
already on the device and are never staged per block. That fixed n*m + m*m cost
must be taken off the budget up front (the same class of fix already applied to
stream_gemm_right's resident output and stream_gemm_left_t's resident
accumulator); relying solely on a live free-memory reading fails on MPS, where
free_compute_bytes() is a static ceiling that never reflects it at all.

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

    def spy(n, cols, backend, item_bytes, frac=subspace._DEFAULT_ROW_BLOCK_FRACTION,
            **kw):
        captured.append(item_bytes)
        return real(n, cols, backend, item_bytes, frac, **kw)

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


def _capture_row_block_kwargs(run):
    """Like _capture_row_block_item_bytes, but records the full kwargs
    (specifically fixed_bytes) rather than just item_bytes."""
    captured = []
    real = subspace._row_block

    def spy(n, cols, backend, item_bytes, frac=subspace._DEFAULT_ROW_BLOCK_FRACTION,
            **kw):
        captured.append({"n": n, "cols": cols, "item_bytes": item_bytes, **kw})
        return real(n, cols, backend, item_bytes, frac, **kw)

    subspace._row_block = spy
    try:
        run()
    finally:
        subspace._row_block = real
    return captured


def test_reconstruct_charges_q_and_ctil_as_fixed_bytes():
    # Q (n, m) and Ctil (m, m) are resident for the whole loop and never
    # shrink with the block -- they must be charged up front, not left for a
    # live free-memory reading that (on MPS) never reflects them at all.
    n, m = 16, 4
    captured = _capture_row_block_kwargs(lambda: _reconstruct(np.float32, np.float32))
    assert captured, "reconstruct did not call _row_block"
    item = np.dtype(np.float32).itemsize
    assert captured[-1]["fixed_bytes"] == (n * m + m * m) * item


def test_reconstruct_old_model_would_have_overshot():
    """Regression witness: ignoring Q/Ctil's residency lets the block size
    exceed the real budget once they're counted."""
    n, m = 4096, 256
    item = np.dtype(np.float32).itemsize
    free = 64 * 1024**2
    frac = 0.3
    budget = int(free * frac)
    resident = (n * m + m * m) * item

    # Pre-fix: _row_block sees none of Q/Ctil's cost, so it hands back a block
    # sized as if the whole budget were free for streaming alone.
    old_blk = subspace._row_block(n, n, _FakeBackend(free), item, frac, out_cols=m)
    old_actual = old_blk * (n + m) * item + resident
    assert old_actual > budget

    # Post-fix: the resident cost is taken off the top first, so the smaller
    # block plus the resident cost together stay within the real budget.
    new_blk = subspace._row_block(n, n, _FakeBackend(free), item, frac,
                                  out_cols=m, fixed_bytes=resident)
    assert new_blk < old_blk
    new_actual = new_blk * (n + m) * item + resident
    assert new_actual <= budget


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
