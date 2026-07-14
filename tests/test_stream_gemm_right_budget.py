"""CPU-only tests for stream_gemm_right's device budget.

stream_gemm_right allocates a full (n, m) output tensor on the device
(``out = xp.empty((n, m))``) that stays resident for the whole streaming loop,
then writes into it one row-block at a time. That resident output is a fixed
cost -- it does not scale with the block -- yet the block size must leave room
for it alongside each staged (blk, cols) input and the (blk, m) GEMM output.

Before the fix, `_row_block` was called with no ``fixed_bytes``, so the block
was sized against the *whole* budget and the resident (n, m) output was never
charged -- under-counting device use by n*m (the entire budget at M = N) and
risking OOM. This mirrors the resident-accumulator accounting stream_gemm_left_t
already does, and the #95 / #138 GEMM-output fixes. Pure arithmetic + a small
recording backend; no GPU needed.

Run:  python tests/test_stream_gemm_right_budget.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace
from strategy.cpu_backend import CPUBackend

ITEM = 4  # fp32
FRAC = subspace._DEFAULT_ROW_BLOCK_FRACTION


class _FakeBackend:
    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free


class _RecordingBackend(CPUBackend):
    """Real CPU compute, but a fixed free-memory budget and a record of the
    largest row-block ever staged to the device."""

    def __init__(self, free_bytes: int):
        super().__init__(verbose=False)
        self._free = free_bytes
        self.max_block = 0

    def free_compute_bytes(self) -> int:
        return self._free

    def to_device(self, host_array):
        self.max_block = max(self.max_block, host_array.shape[0])
        return super().to_device(host_array)


def test_row_block_charges_the_resident_output_up_front():
    """With the (n, m) output charged as fixed_bytes, staged rows + per-row GEMM
    output + the resident output all fit; the old (no fixed_bytes) call overshoots.

    Regime: the resident output fits in the budget (as it must -- it is unavoidable
    device memory), but streaming the rows at full height would push past it. The
    fix is what shrinks the block to leave room for the resident output."""
    n, m = 4096, 1024
    free = 80 * 1024**2
    budget = int(free * FRAC)
    bk = _FakeBackend(free)

    fixed = n * m * ITEM
    blk = subspace._row_block(n, n, bk, ITEM, FRAC, out_cols=m, fixed_bytes=fixed)
    peak = fixed + blk * (n + m) * ITEM     # resident out + staged(blk,n) + temp(blk,m)
    assert peak <= budget

    old_blk = subspace._row_block(n, n, bk, ITEM, FRAC, out_cols=m)  # no fixed_bytes
    old_peak = fixed + old_blk * (n + m) * ITEM
    assert old_peak > budget                # the resident output was unbudgeted


def test_stream_gemm_right_block_leaves_room_for_the_resident_output():
    """Integration: run the real primitive and confirm the block it actually
    stages, plus the resident (n, m) output, stays within the VRAM budget."""
    n, m = 512, 512
    # Pick a budget that forces genuine blocking (blk < n) so the resident output
    # is not trivially dominated by a single full-height pass.
    free = 4_177_920
    budget = int(free * FRAC)
    bk = _RecordingBackend(free)

    X = np.ones((n, n), dtype=np.float32)
    Q = np.ones((n, m), dtype=np.float32)
    subspace.stream_gemm_right(X, Q, bk, np.float32)

    blk = bk.max_block
    assert 1 <= blk < n                     # actually blocked
    peak = (n * m + blk * (n + m)) * ITEM   # resident out + staged + per-row temp
    assert peak <= budget


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
