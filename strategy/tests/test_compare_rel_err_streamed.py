"""CPU-only tests for compare()'s streamed relative-Frobenius error (issue #166).

strategy.runner.compare reported rel_err by casting BOTH full (n, n) products to
float64 at once -- which force-loads disk-backed memmaps into host RAM (~3*n^2*8
bytes) and OOMs at the out-of-core sizes --compare targets. The streamed helper
must match the naive result exactly while only ever holding one float64 row-block
of each operand. Pure NumPy; no GPU needed.

Run:  python strategy/tests/test_compare_rel_err_streamed.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import storage
from strategy.runner import _rel_frobenius_streamed


def _naive(Ce, Cs):
    ref = np.asarray(Ce, dtype=np.float64)
    return float(np.linalg.norm(np.asarray(Cs, dtype=np.float64) - ref)
                 / (np.linalg.norm(ref) or 1.0))


def test_matches_naive_in_ram():
    rng = np.random.default_rng(0)
    n = 128
    Ce = rng.standard_normal((n, n)).astype(np.float32)
    Cs = Ce + 0.01 * rng.standard_normal((n, n)).astype(np.float32)
    got = _rel_frobenius_streamed(Ce, Cs)
    assert abs(got - _naive(Ce, Cs)) < 1e-12


def test_block_size_invariant():
    """The result must not depend on how the rows are chunked."""
    rng = np.random.default_rng(1)
    n = 100
    Ce = rng.standard_normal((n, n)).astype(np.float32)
    Cs = rng.standard_normal((n, n)).astype(np.float32)
    ref = _rel_frobenius_streamed(Ce, Cs)
    # force blk = 1, a few rows, all rows, more than n
    for bb in (8, n * 8 * 3, n * n * 8 * 10):
        assert abs(_rel_frobenius_streamed(Ce, Cs, block_bytes=bb) - ref) < 1e-12
    # tiny budget still streams (blk clamps to >= 1) and stays correct
    assert abs(_rel_frobenius_streamed(Ce, Cs, block_bytes=1) - ref) < 1e-12


def test_identical_products_zero_error():
    rng = np.random.default_rng(2)
    C = rng.standard_normal((64, 64)).astype(np.float32)
    assert _rel_frobenius_streamed(C, C.copy()) == 0.0


def test_zero_reference_does_not_divide_by_zero():
    z = np.zeros((16, 16), dtype=np.float32)
    nz = np.ones((16, 16), dtype=np.float32)
    # den == 0 -> guarded to 1.0, so this is finite, not inf/nan
    out = _rel_frobenius_streamed(z, nz)
    assert np.isfinite(out) and out > 0.0


def test_works_on_disk_memmaps_blockwise():
    """The real scenario: disk-backed memmaps, streamed a block at a time.

    Spy on np.asarray to prove no single call ever materializes the full matrix."""
    workdir = tempfile.mkdtemp(prefix="cco_compare_test_")
    try:
        n = 96
        pe = os.path.join(workdir, "Ce.dat")
        ps = os.path.join(workdir, "Cs.dat")
        Ce = storage.allocate(n, np.float32, True, pe)
        Cs = storage.allocate(n, np.float32, True, ps)
        rng = np.random.default_rng(3)
        Ce[:] = rng.standard_normal((n, n)).astype(np.float32)
        Cs[:] = Ce + 0.05 * rng.standard_normal((n, n)).astype(np.float32)
        Ce.flush(); Cs.flush()

        max_rows = {"n": 0}
        real_asarray = np.asarray

        def spy(a, *args, **kw):
            arr = real_asarray(a, *args, **kw)
            if getattr(arr, "ndim", 0) == 2:
                max_rows["n"] = max(max_rows["n"], arr.shape[0])
            return arr

        np.asarray = spy
        try:
            # a small block budget forces multiple blocks
            got = _rel_frobenius_streamed(Ce, Cs, block_bytes=n * 8 * 10)
        finally:
            np.asarray = real_asarray

        # reference computed the plain way (in RAM, small n)
        ref = _naive(np.array(Ce), np.array(Cs))
        assert abs(got - ref) < 1e-9
        # never loaded all n rows at once
        assert 0 < max_rows["n"] < n, max_rows["n"]
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


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
