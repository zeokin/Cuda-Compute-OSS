"""CPU-only tests for subspace._row_block's device budget (issue #138).

A streamed row-block costs more than the rows it stages: each iteration also
allocates the GEMM output, which cannot alias its operands and is live at the
same time. `_row_block` must budget those too, or the block overshoots
`vram_fraction x free` -- up to 2x at M = N -- and can OOM. This mirrors the
accounting `matmul/gemm.py` adopted in #95. Pure arithmetic; no GPU needed.

Run:  python tests/test_row_block_budget.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace


class _FakeBackend:
    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free


FREE = 64 * 1024**2
FRAC = 0.3
ITEM = 4  # fp32


def _budget(free=FREE, frac=FRAC):
    return int(free * frac)


def test_transient_reallocated_input_tile_is_budgeted():
    """The staged (blk, cols) input is reallocated every iteration
    (``Xr = backend.to_device(...)``): the new tile is built before the name
    rebinds and the old one is dropped, so two are briefly live at once.
    ``_row_block`` must budget ``2 * cols`` per row even with no GEMM output --
    mirroring matmul/gemm.py's transient-fifth-tile term (#4afca0e)."""
    n = 4096
    bk = _FakeBackend(FREE)
    blk = subspace._row_block(n, n, bk, ITEM, FRAC)          # out_cols=0
    assert 2 * blk * n * ITEM <= _budget()                    # both staged tiles fit
    # Charging only one staged tile (the pre-fix model) would have doubled it.
    naive = min(n, max(1, _budget() // (n * ITEM)))
    assert blk < naive
    # For out_cols <= cols the transient dominates, so naming an output no wider
    # than the staged input does not shrink the block further; only a WIDER
    # output does.
    assert subspace._row_block(n, n, bk, ITEM, FRAC, out_cols=n) == blk
    assert subspace._row_block(n, n, bk, ITEM, FRAC, out_cols=2 * n) < blk


def test_block_stays_within_budget_at_m_equals_n():
    """M = N is the exactness path; staged rows + GEMM output must both fit."""
    n, m = 4096, 4096
    blk = subspace._row_block(n, n, _FakeBackend(FREE), ITEM, FRAC, out_cols=m)
    actual = blk * (n + m) * ITEM          # staged (blk,n) + output (blk,m)
    assert actual <= _budget()


def test_block_stays_within_budget_at_default_m():
    n, m = 4096, 4096 // 8
    blk = subspace._row_block(n, n, _FakeBackend(FREE), ITEM, FRAC, out_cols=m)
    actual = blk * (n + max(n, m)) * ITEM     # staged + transient input (m < n)
    assert actual <= _budget()


def test_pre_transient_model_would_have_overshot():
    """Regression witness for the transient term: for M < N the reallocated
    input tile (2n) is a bigger peak than staged+output (n+m), so a block sized
    by the old ``n + m`` model overshoots the true peak at the reallocation."""
    n, m = 4096, 512
    budget = _budget()
    old_blk = min(n, max(1, budget // ((n + m) * ITEM)))   # pre-fix: staged + output
    true_peak = old_blk * (n + max(n, m)) * ITEM           # staged + transient input
    assert true_peak > budget
    assert true_peak / budget > 1.5                        # n = 8m -> ~1.8x


def test_fixed_bytes_is_taken_off_the_budget():
    """stream_gemm_left_t's (n, m) product does not scale with the block, so it
    is charged up front rather than per row."""
    n, m = 1024, 256
    bk = _FakeBackend(FREE)
    fixed = n * m * ITEM
    blk = subspace._row_block(n, n, bk, ITEM, FRAC, fixed_bytes=fixed)
    assert blk * n * ITEM + fixed <= _budget()
    # and it must be no larger than the unconstrained block
    assert blk <= subspace._row_block(n, n, bk, ITEM, FRAC)


def test_block_never_drops_below_one():
    """Even when the fixed cost swallows the whole budget, stream at least one
    row rather than returning 0 (which would make the loop spin forever)."""
    n, m = 4096, 4096
    blk = subspace._row_block(n, n, _FakeBackend(1024), ITEM, FRAC,
                              out_cols=m, fixed_bytes=10**9)
    assert blk >= 1


def test_defaults_preserve_previous_behavior():
    """out_cols=0, fixed_bytes=0 reproduces the original formula exactly."""
    n, cols = 512, 512
    bk = _FakeBackend(FREE)
    expected = min(n, max(1, _budget() // (cols * ITEM)))
    assert subspace._row_block(n, cols, bk, ITEM, FRAC) == expected


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
