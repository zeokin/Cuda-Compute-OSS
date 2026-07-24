"""Matrix generation must stay inside its own ~256 MiB per-block budget.

``_fill_random`` and ``_fill_iota`` both stream row-blocks sized so one block is
~256 MiB of fp64/int64 temporaries -- that block loop is the only thing keeping
the out-of-core (128k x 128k) regime off the host's RAM. Both then built the
block with an out-of-place operation:

    chunk = rng.standard_normal((rows, n)) * scale      # draw AND scaled copy
    mat[r0:r1, :] = ((rows + cols) % 97).astype(...)    # sum  AND remainder

so the operand stayed live alongside its result and one block actually cost two
-- exactly the pattern #297 already fixed for ``_fill_lowrank`` /
``_fill_decaying_spectrum``'s V, left behind in these two. ``random`` is the
default fill for the CLIs and for ``python -m eval``.

Both packages carry their own copy of storage.py, so this is a parity table:
the rule cannot hold in one and silently lapse in the other.

Pure CPU, no GPU, no torch.

Run:  python tests/test_fill_host_peak.py
"""
from __future__ import annotations

import gc
import os
import sys
import tracemalloc

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul import storage as matmul_storage
from strategy import storage as strategy_storage

STORAGES = [("matmul", matmul_storage), ("strategy", strategy_storage)]

# Small enough to keep the suite fast, large enough that one row-block dominates
# any interpreter noise tracemalloc also sees.
N = 1024
BLOCK_BYTES = N * N * 8          # block == N at this size, int64/fp64 elements

# The old out-of-place forms peaked at 2.0x one block; in-place peaks at 1.5x
# (block + the fp32 cast on the way into `mat`). Sits clear of both.
MAX_PEAK_RATIO = 1.75


def _block_rows(n: int) -> int:
    """The row-block both fills compute -- kept in sync with storage.py."""
    return max(1, min(n, (256 * 1024**2) // (n * 8)))


def _peak_bytes(fn) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def _measure(fill, dtype=np.float32) -> float:
    """Peak traced bytes for one fill, as a multiple of one row-block."""
    mat = np.empty((N, N), dtype=dtype)
    peak = _peak_bytes(lambda: fill(mat, 0))
    if peak < BLOCK_BYTES // 2:
        # numpy allocations are invisible to tracemalloc on this build; the
        # measurement means nothing, so don't assert on it.
        pytest.skip("tracemalloc does not trace numpy allocations here")
    return peak / BLOCK_BYTES


def test_block_rows_assumption_holds():
    # The whole measurement assumes block == N at this N, i.e. one iteration.
    assert _block_rows(N) == N


@pytest.mark.parametrize("label,storage", STORAGES)
def test_fill_random_stays_within_one_block(label, storage):
    ratio = _measure(storage._fill_random)
    assert ratio < MAX_PEAK_RATIO, f"{label}: _fill_random peaked at {ratio:.2f}x one block"


@pytest.mark.parametrize("label,storage", STORAGES)
def test_fill_iota_stays_within_one_block(label, storage):
    ratio = _measure(storage._fill_iota)
    assert ratio < MAX_PEAK_RATIO, f"{label}: _fill_iota peaked at {ratio:.2f}x one block"


@pytest.mark.parametrize("label,storage", STORAGES)
def test_fill_lowrank_reference_stays_within_budget(label, storage):
    # Only strategy carries the low-rank fills; matmul's storage has none. This
    # is the already-fixed (#297) reference point the two above must match.
    fill = getattr(storage, "_fill_lowrank", None)
    if fill is None:
        pytest.skip(f"{label} has no low-rank fill")
    mat = np.empty((N, N), dtype=np.float32)
    peak = _peak_bytes(lambda: fill(mat, 0, 16))
    if peak < BLOCK_BYTES // 2:
        pytest.skip("tracemalloc does not trace numpy allocations here")
    assert peak / BLOCK_BYTES < MAX_PEAK_RATIO, label


class _SpyRNG:
    """Hands back a caller-owned buffer for the row-block draw, so the test can
    see whether the fill scaled it in place or built a second array."""

    def __init__(self, draw: np.ndarray):
        self.draw = draw

    def standard_normal(self, size):
        assert tuple(size) == self.draw.shape
        return self.draw


@pytest.mark.parametrize("label,storage", STORAGES)
def test_fill_random_scales_the_draw_in_place(label, storage):
    # Deterministic companion to the measurement: an out-of-place ``* scale``
    # leaves the returned draw untouched.
    n, scale = 8, 0.25
    draw = np.ones((n, n), dtype=np.float64)
    real = storage.np.random.default_rng
    storage.np.random.default_rng = lambda seed: _SpyRNG(draw)
    try:
        storage._fill_random(np.empty((n, n), dtype=np.float32), seed=0, scale=scale)
    finally:
        storage.np.random.default_rng = real
    assert np.allclose(draw, np.full((n, n), scale)), label


# --- the fix must not change a single generated value ----------------------
@pytest.mark.parametrize("label,storage", STORAGES)
def test_fill_random_values_are_unchanged(label, storage):
    n, scale = 64, 0.75
    got = np.empty((n, n), dtype=np.float64)
    storage._fill_random(got, seed=7, scale=scale)
    want = np.random.default_rng(7).standard_normal((n, n)) * scale
    assert np.array_equal(got, want), label


@pytest.mark.parametrize("label,storage", STORAGES)
def test_fill_iota_values_are_unchanged(label, storage):
    n = 64
    got = np.empty((n, n), dtype=np.float32)
    storage._fill_iota(got, seed=7)
    rng = np.random.default_rng(7)
    row_shift = rng.integers(0, 97, size=n)
    col_shift = rng.integers(0, 97, size=n)
    want = (((np.arange(n) + row_shift)[:, None] + (np.arange(n) + col_shift)) % 97)
    assert np.array_equal(got, want.astype(np.float32)), label


@pytest.mark.parametrize("label,storage", STORAGES)
def test_fills_still_work_on_a_memmap(tmp_path, label, storage):
    # The block loop exists for disk-backed matrices; in-place ops must not be
    # applied to the memmap itself, only to the host-side block.
    n = 32
    path = tmp_path / f"{label}.dat"
    for fill in (storage._fill_random, storage._fill_iota):
        mat = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, n))
        fill(mat, 3)
        assert np.isfinite(np.asarray(mat)).all(), label
        assert np.asarray(mat).any(), label
        del mat


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
