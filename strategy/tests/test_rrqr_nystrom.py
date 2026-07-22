"""CPU-safe tests for the built-in ``rrqr-nystrom`` transform.

No GPU required — uses the smoke-test CPUBackend so these always run in the
contributor CI path (`pytest strategy/tests`).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy.cpu_backend import CPUBackend
from strategy.storage import generate
from strategy.transforms import (
    NystromTransform,
    RRQRNystromTransform,
    available,
    get_transform,
)


def test_rrqr_nystrom_registered():
    assert "rrqr-nystrom" in available()
    assert isinstance(get_transform("rrqr-nystrom"), RRQRNystromTransform)


def test_rrqr_nystrom_requires_operands():
    backend = CPUBackend(verbose=False)
    t = RRQRNystromTransform(seed=0)
    try:
        t.basis(32, 8, backend, np.float32, A=None, B=None)
        assert False, "expected ValueError when A/B missing"
    except ValueError as e:
        assert "rrqr-nystrom" in str(e)


def test_rrqr_nystrom_rejects_bad_m():
    backend = CPUBackend(verbose=False)
    t = RRQRNystromTransform(seed=0)
    A = generate(16, np.float32, False, None, seed=1, fill="random")
    B = generate(16, np.float32, False, None, seed=2, fill="random")
    for bad_m in (0, 17):
        try:
            t.basis(16, bad_m, backend, np.float32, A=A, B=B)
            assert False, f"expected ValueError for m={bad_m}"
        except ValueError as e:
            assert "rrqr-nystrom" in str(e)


def test_rrqr_nystrom_basis_shape_orthonormal():
    backend = CPUBackend(verbose=False)
    n, m = 64, 18
    A = generate(n, np.float32, False, None, seed=1, fill="lowrank", data_rank=4)
    B = generate(n, np.float32, False, None, seed=2, fill="lowrank", data_rank=4)
    Q = RRQRNystromTransform(seed=0).basis(n, m, backend, np.float32, A=A, B=B)
    Qh = backend.to_host(Q).astype(np.float64)

    assert Qh.shape == (n, m)
    assert np.isfinite(Qh).all()
    gram = Qh.T @ Qh
    assert float(np.linalg.norm(gram - np.eye(m))) < 1e-5


def test_rrqr_nystrom_seed_deterministic():
    backend = CPUBackend(verbose=False)
    n, m = 48, 12
    A = generate(n, np.float32, False, None, seed=3, fill="random")
    B = generate(n, np.float32, False, None, seed=4, fill="random")
    Q1 = backend.to_host(RRQRNystromTransform(seed=7).basis(n, m, backend, np.float32, A, B))
    Q2 = backend.to_host(RRQRNystromTransform(seed=7).basis(n, m, backend, np.float32, A, B))
    Q3 = backend.to_host(RRQRNystromTransform(seed=8).basis(n, m, backend, np.float32, A, B))
    assert np.allclose(Q1, Q2)
    assert not np.allclose(Q1, Q3)


def test_rrqr_nystrom_recovers_low_rank_product():
    backend = CPUBackend(verbose=False)
    n, r, m = 96, 6, 30  # M > 3r, comfortably enough budget for exact recovery
    A = generate(n, np.float64, False, None, seed=11, fill="lowrank", data_rank=r)
    B = generate(n, np.float64, False, None, seed=12, fill="lowrank", data_rank=r)
    Q = RRQRNystromTransform(seed=0).basis(n, m, backend, np.float64, A=A, B=B)
    Qh = backend.to_host(Q)

    P = Qh @ Qh.T
    C_exact = A @ B
    C_approx = P @ A @ P @ B @ P
    rel_err = np.linalg.norm(C_approx - C_exact) / np.linalg.norm(C_exact)
    assert rel_err < 1e-6


def test_rrqr_nystrom_basis_flops_honest():
    n, m = 12000, 1500
    rr = RRQRNystromTransform()  # default oversample=4
    ny = NystromTransform().basis_flops(n, m)
    assert rr.basis_flops(n, m) > ny  # the pivoted selection pass is not free
    assert rr.basis_flops(n, m) > 0.0


def test_rrqr_nystrom_oversample_one_still_valid():
    # oversample=1 means the candidate pool IS the final width -- the pivoted
    # selection loop must degrade gracefully (nothing left to choose between)
    # rather than erroring.
    backend = CPUBackend(verbose=False)
    n, m = 64, 18
    A = generate(n, np.float32, False, None, seed=1, fill="lowrank", data_rank=4)
    B = generate(n, np.float32, False, None, seed=2, fill="lowrank", data_rank=4)
    t = RRQRNystromTransform(seed=0)
    t.oversample = 1
    Q = backend.to_host(t.basis(n, m, backend, np.float32, A=A, B=B))
    assert Q.shape == (n, m)
    assert np.isfinite(Q).all()


def test_rrqr_nystrom_beats_plain_nystrom_on_decaying_spectrum():
    # Column-space recovery only (isolating the selection mechanism itself,
    # the same check used to validate the idea before implementing it): a
    # pivoted, redundancy-avoiding landmark pick from an oversampled pool
    # should span col(A) better than an equal-size uniform-random pick, even
    # though this data's marginal per-column importance is flat (ruling out
    # leverage-score weighting is what motivated this transform instead).
    n, rank, w = 512, 200, 60
    A = generate(n, np.float64, False, None, seed=21, fill="decaying-spectrum",
                 data_rank=rank, spectral_alpha=1.0)

    def col_space_rel_err(idx):
        Q, _ = np.linalg.qr(A[:, idx])
        P = Q @ Q.T
        return np.linalg.norm(P @ A - A) / np.linalg.norm(A)

    rng = np.random.default_rng(1000)
    idx_uniform = rng.choice(n, size=w, replace=False)
    err_uniform = col_space_rel_err(idx_uniform)

    cand_idx = rng.choice(n, size=4 * w, replace=False)
    from strategy.transforms import _pivoted_select
    sel = _pivoted_select(A[:, cand_idx], w)
    err_pivoted = col_space_rel_err(cand_idx[sel])

    assert err_pivoted < err_uniform


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
