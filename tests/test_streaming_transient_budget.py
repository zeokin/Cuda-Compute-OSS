"""CPU-only tests for the reassigned-tile transient budget (issue #234).

The strategy streaming primitives reassign a named (rb, n) device tensor every
row-block (``Xr = to_device(...)`` / ``outr = matmul(...)``). Python allocates the
new tile before dropping the old one, and PyTorch's caching allocator keeps both
blocks reserved, so the reassigned tile's momentary duplicate must be budgeted --
the streaming analog of the tiled engine's transient-tile fix in matmul/gemm.py.

These tests assert _row_block honors ``transient_cols`` and that each primitive
passes it, and that the streamed math is unchanged. Pure NumPy; no GPU needed.

Run:  python tests/test_streaming_transient_budget.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace
from strategy.cpu_backend import CPUBackend

FRAC = subspace._DEFAULT_ROW_BLOCK_FRACTION


class _Static:
    """Fixed free-memory (MPS-like static ceiling)."""

    def __init__(self, free_bytes):
        self._free = free_bytes

    def free_compute_bytes(self):
        return self._free


def test_row_block_charges_transient_cols():
    n, cols, item, m = 8192, 8192, 4, 512
    bk = _Static(200 * 1024**2)
    without = subspace._row_block(n, cols, bk, item, FRAC, out_cols=m)
    witht = subspace._row_block(n, cols, bk, item, FRAC, out_cols=m, transient_cols=cols)
    assert witht < without
    # per_row grew by exactly transient_cols: (cols+out_cols) -> (2*cols+out_cols)
    budget = int(bk.free_compute_bytes() * FRAC)
    assert witht == min(n, max(1, budget // ((cols + m + cols) * item)))


def _captured_kwargs(run):
    seen = []
    real = subspace._row_block

    def spy(bn, cols, backend, item_bytes, frac=FRAC, out_cols=0, fixed_bytes=0,
            transient_cols=0):
        seen.append({"cols": cols, "out_cols": out_cols, "transient_cols": transient_cols})
        return real(bn, cols, backend, item_bytes, frac, out_cols=out_cols,
                    fixed_bytes=fixed_bytes, transient_cols=transient_cols)

    subspace._row_block = spy
    try:
        run()
    finally:
        subspace._row_block = real
    return seen


def test_each_primitive_passes_the_reassigned_tile_width():
    n, m = 64, 16
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, n))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    Ctil = rng.standard_normal((m, m))
    bk = CPUBackend(verbose=False)

    for name, run in [
        ("stream_gemm_right", lambda: subspace.stream_gemm_right(X, Q, bk, np.float64)),
        ("stream_gemm_left_t", lambda: subspace.stream_gemm_left_t(X, Q, bk, np.float64)),
        ("compress", lambda: subspace.compress(X, Q, bk, np.float64)),
        ("reconstruct", lambda: subspace.reconstruct(
            Ctil, Q, np.zeros((n, n)), bk, np.float64, compute_dtype=np.float64)),
    ]:
        seen = _captured_kwargs(run)
        assert seen, name
        # the reassigned (rb, n) tile is n-wide, so transient_cols must be n
        assert seen[-1]["transient_cols"] == n, (name, seen[-1])


def test_streamed_math_unchanged_multi_block():
    """The budget change must not alter results, even when blocking is forced."""
    class Tight(CPUBackend):
        def free_compute_bytes(self):
            return 150_000

    n, m = 96, 12
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, n))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    Ctil = rng.standard_normal((m, m))
    bk = Tight(verbose=False)

    def rel(a, b):
        return np.linalg.norm(np.asarray(a) - b) / (np.linalg.norm(b) or 1.0)

    assert rel(bk.to_host(subspace.stream_gemm_right(X, Q, bk, np.float64)), X @ Q) < 1e-12
    assert rel(bk.to_host(subspace.stream_gemm_left_t(X, Q, bk, np.float64)), X.T @ Q) < 1e-12
    assert rel(bk.to_host(subspace.compress(X, Q, bk, np.float64)), Q.T @ X @ Q) < 1e-12
    C = np.zeros((n, n))
    subspace.reconstruct(Ctil, Q, C, bk, np.float64, compute_dtype=np.float64)
    assert rel(C, Q @ Ctil @ Q.T) < 1e-12


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
