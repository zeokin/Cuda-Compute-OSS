"""CPU-only tests for stream_gemm_left_t's device budget.

stream_gemm_left_t computes X^T @ Q by summing per-row-block products into a
resident (n, m) accumulator:

    acc = xp.zeros((n, m))                 # resident (n, m) DEVICE buffer
    for rb: acc += matmul(Xr.T, Q[rb])     # a second (n, m) product each step

At the peak of ``acc += matmul(...)`` TWO (n, m) buffers are live -- ``acc`` and
the freshly allocated product, which cannot alias it -- neither scaling with the
block. The budget must charge both (2*n*m). Charging only the product (n*m)
leaves the accumulator unbudgeted, sizes the block against the whole budget and
can OOM. This mirrors stream_gemm_right's resident-output fix and #138. Pure
arithmetic + a small recording backend; no GPU needed.

Run:  python tests/test_stream_gemm_left_t_budget.py
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
    """Real CPU compute, fixed free-memory budget, records the largest staged block."""

    def __init__(self, free_bytes: int):
        super().__init__(verbose=False)
        self._free = free_bytes
        self.max_block = 0

    def free_compute_bytes(self) -> int:
        return self._free

    def to_device(self, host_array):
        self.max_block = max(self.max_block, host_array.shape[0])
        return super().to_device(host_array)


def test_row_block_charges_both_resident_nm_buffers():
    """acc + the per-step product are both (n, m); charging 2*n*m keeps the peak
    within budget, while charging only n*m (the old model) overshoots."""
    n, m = 2048, 512
    free = 40 * 1024**2
    budget = int(free * FRAC)
    bk = _FakeBackend(free)

    blk = subspace._row_block(n, n, bk, ITEM, FRAC, fixed_bytes=2 * n * m * ITEM)
    peak = 2 * n * m * ITEM + blk * n * ITEM        # acc + product + staged (blk, n)
    assert peak <= budget

    old_blk = subspace._row_block(n, n, bk, ITEM, FRAC, fixed_bytes=n * m * ITEM)
    old_peak = 2 * n * m * ITEM + old_blk * n * ITEM  # true peak still has BOTH (n,m)
    assert old_peak > budget                         # accumulator was unbudgeted


def test_stream_gemm_left_t_block_leaves_room_for_accumulator_and_product():
    """Integration: run the real primitive and confirm the block it stages, plus
    both resident (n, m) buffers, stays within the VRAM budget."""
    n, m = 512, 128
    free = 2_430_293                      # forces genuine blocking (blk < n)
    budget = int(free * FRAC)
    bk = _RecordingBackend(free)

    X = np.ones((n, n), dtype=np.float32)
    Q = np.ones((n, m), dtype=np.float32)
    subspace.stream_gemm_left_t(X, Q, bk, np.float32)

    blk = bk.max_block
    assert 1 <= blk < n
    peak = (2 * n * m + blk * n) * ITEM   # acc + product + staged (blk, n)
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
