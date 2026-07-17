"""Every streaming primitive must charge its resident Q operand (issue #236).

``reconstruct`` already takes Q (n, m) and Ctil (m, m) off the row-block budget
up front, and says why in its own comment: they "arrive already on the device and
are never staged per block", so they are a fixed cost that "must be taken off the
budget up front ... rather than left to a live free-memory reading that, on MPS,
is a static ceiling and never reflects it at all."

``stream_gemm_right``, ``stream_gemm_left_t`` and ``compress`` take the same
resident Q, but charged only their own buffers -- so each under-budgeted by n*m,
the term that dominates once n > m (the regime the subspace strategy targets).

This is a parity table: every primitive is driven through the SAME check, so the
resident-Q rule cannot hold for one and silently lapse in another. Spying on
_row_block and running on a CPU backend pins the contract with no GPU.

Run:  python tests/test_streaming_q_operand_budget.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import subspace
from strategy.cpu_backend import CPUBackend

FRAC = subspace._DEFAULT_ROW_BLOCK_FRACTION
DTYPE = np.float32
ITEM = np.dtype(DTYPE).itemsize


def _fixed_bytes_of(call) -> int:
    """Run ``call`` with _row_block spied; return the fixed_bytes it was handed."""
    seen = {}
    real = subspace._row_block

    def spy(bn, cols, backend, item_bytes, frac=FRAC, out_cols=0, fixed_bytes=0, **kw):
        seen.setdefault("fixed_bytes", fixed_bytes)
        return real(bn, cols, backend, item_bytes, frac,
                    out_cols=out_cols, fixed_bytes=fixed_bytes, **kw)

    subspace._row_block = spy
    try:
        call()
    finally:
        subspace._row_block = real
    return seen["fixed_bytes"]


def _drivers(n, m):
    """(label, thunk, expected_fixed_bytes) for every primitive that holds Q."""
    backend = CPUBackend(verbose=False)
    X = np.zeros((n, n), dtype=DTYPE)
    Q = np.zeros((n, m), dtype=DTYPE)
    Ctil = np.zeros((m, m), dtype=DTYPE)
    C_out = np.zeros((n, n), dtype=DTYPE)
    return [
        ("stream_gemm_right",
         lambda: subspace.stream_gemm_right(X, Q, backend, DTYPE),
         2 * n * m * ITEM),                          # out + Q
        ("stream_gemm_left_t",
         lambda: subspace.stream_gemm_left_t(X, Q, backend, DTYPE),
         3 * n * m * ITEM),                          # acc + product + Q
        ("compress",
         lambda: subspace.compress(X, Q, backend, DTYPE),
         (n * m + 2 * m * m) * ITEM),                # Q + acc + product
        ("reconstruct",
         lambda: subspace.reconstruct(Ctil, Q, C_out, backend, DTYPE),
         (n * m + m * m) * ITEM),                    # Q + Ctil (already correct)
    ]


@pytest.mark.parametrize("n,m", [(64, 16), (128, 32), (256, 8)], ids=repr)
def test_every_primitive_charges_its_resident_q(n, m):
    for label, thunk, expected in _drivers(n, m):
        got = _fixed_bytes_of(thunk)
        assert got == expected, f"{label}: fixed_bytes={got}, expected {expected}"


@pytest.mark.parametrize("n,m", [(64, 16), (512, 4)], ids=repr)
def test_dropping_q_would_under_budget_by_exactly_n_times_m(n, m):
    """The regression this guards: each charge must exceed its Q-less predecessor
    by exactly n*m -- so a future edit that quietly drops Q is caught, not just a
    wrong total."""
    q_bytes = n * m * ITEM
    without_q = {
        "stream_gemm_right": n * m * ITEM,               # out only
        "stream_gemm_left_t": 2 * n * m * ITEM,          # acc + product only
        "compress": 2 * m * m * ITEM,                    # acc + product only
        "reconstruct": m * m * ITEM,                     # Ctil only
    }
    for label, thunk, _expected in _drivers(n, m):
        got = _fixed_bytes_of(thunk)
        assert got - without_q[label] == q_bytes, (
            f"{label}: Q ({q_bytes} B) is not charged"
        )


def test_extra_fixed_bytes_still_adds_on_top_of_q():
    """rsvd passes earlier sketch parts as extra_fixed_bytes (#265); charging Q
    must not displace that -- the two are additive."""
    n, m, extra = 64, 16, 4096
    backend = CPUBackend(verbose=False)
    X = np.zeros((n, n), dtype=DTYPE)
    Q = np.zeros((n, m), dtype=DTYPE)

    base = _fixed_bytes_of(lambda: subspace.stream_gemm_right(X, Q, backend, DTYPE))
    with_extra = _fixed_bytes_of(
        lambda: subspace.stream_gemm_right(X, Q, backend, DTYPE, extra_fixed_bytes=extra)
    )
    assert with_extra - base == extra


def test_streamed_math_is_unchanged_under_a_tight_budget():
    """Charging more can only shrink the block; the results must be identical."""
    class Tight(CPUBackend):
        def free_compute_bytes(self):
            return 100_000

    n, m = 96, 12
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, n))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    backend = Tight(verbose=False)

    right = np.asarray(subspace.stream_gemm_right(X, Q, backend, np.float64))
    left = np.asarray(subspace.stream_gemm_left_t(X, Q, backend, np.float64))
    comp = np.asarray(subspace.compress(X, Q, backend, np.float64))

    assert np.allclose(right, X @ Q)
    assert np.allclose(left, X.T @ Q)
    assert np.allclose(comp, Q.T @ X @ Q)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
