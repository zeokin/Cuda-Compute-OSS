"""CPU-only tests: rsvd charges the resident assembly buffer into stream budgets.

rsvd.basis preallocates the (n, m) output ``Y`` and streams each sketch into a
column slice (#307). ``Y`` is charged as ``extra_fixed_bytes`` on every sketch
so MPS-static free-memory accounting cannot ignore it. Pure stub/arithmetic;
no GPU needed.

Run:  python strategy/tests/test_rsvd_prior_parts_budget.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import subspace as sub
from strategy.transforms import RandomizedSVDTransform


class _XP:
    empty = staticmethod(np.empty)

    class linalg:
        qr = staticmethod(np.linalg.qr)


class _FakeBackend:
    xp = _XP()

    def to_device(self, x):
        return np.asarray(x)


def _capture_extra_fixed(n=96, m=30, dtype=np.float32):
    """Call rsvd.basis with stubbed sketches; return each call's extra_fixed_bytes."""
    captured = []
    orig_r, orig_l = sub.stream_gemm_right, sub.stream_gemm_left_t

    def fake(X, Q, backend, dt, frac=sub._DEFAULT_ROW_BLOCK_FRACTION,
             extra_fixed_bytes=0):
        captured.append(int(extra_fixed_bytes))
        return np.zeros((X.shape[0], Q.shape[1]), dtype=dt)

    sub.stream_gemm_right = fake
    sub.stream_gemm_left_t = fake
    try:
        A = np.eye(n, dtype=dtype)
        B = np.eye(n, dtype=dtype)
        Q = RandomizedSVDTransform(seed=0).basis(
            n, m, _FakeBackend(), dtype, A=A, B=B, frac=0.3
        )
        assert Q.shape == (n, m)
    finally:
        sub.stream_gemm_right, sub.stream_gemm_left_t = orig_r, orig_l
    return captured


def test_rsvd_charges_assembly_y_on_every_sketch():
    # Y is (n, m) and resident for the whole basis stage — every sketch sees it.
    n, m = 96, 30
    item = np.dtype(np.float32).itemsize
    y_bytes = n * m * item
    seen = _capture_extra_fixed(n=n, m=m)
    assert seen == [y_bytes, y_bytes, y_bytes], seen


def test_rsvd_assembly_charge_keeps_sketch_peak_within_budget():
    """Arithmetic: with Y charged, sketch-3 peak fits; old parts+Y spike does not."""
    n, m, item, frac = 8192, 1024, 4, 0.3
    free = 180 * 1024**2  # tight enough that 2*n*m exceeds frac*free
    budget = int(free * frac)
    base, rem = divmod(m, 3)
    w = [base + (1 if i < rem else 0) for i in range(3)]
    y_bytes = n * m * item
    # left_t steady-state core 2*n*w + Y as extra.
    fixed = 2 * n * w[2] * item + y_bytes
    assert fixed < budget
    blk = max(1, (budget - fixed) // (n * item))
    peak = fixed + blk * n * item
    assert peak <= budget

    # Old model after sketches: parts + concatenate Y = 2*n*m, never charged.
    old_assembly_peak = 2 * n * m * item
    assert old_assembly_peak > budget


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
