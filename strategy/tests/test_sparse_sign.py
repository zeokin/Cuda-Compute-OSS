"""CPU-safe tests for the ``sparse_sign`` (OSNAP-style) transform.

No GPU required -- uses the smoke-test CPUBackend so these always run in the
contributor CI path (`pytest strategy/tests`). Mirrors test_nystrom.py's
structure for the shared registration/validation/orthonormality/determinism/
basis_flops contract every Transform must satisfy, plus a reproducible
regression of the numerical premise-check reported in the transform's
docstring and the PR: sparse_sign measurably beats plain nystrom on this
project's own decaying-spectrum fill, at the same (host-gather, no-GEMM)
cost class.
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
    RandomizedSVDTransform,
    SparseSignTransform,
    available,
    get_transform,
)


def test_sparse_sign_registered():
    assert "sparse_sign" in available()
    assert isinstance(get_transform("sparse_sign"), SparseSignTransform)


def test_sparse_sign_requires_operands():
    backend = CPUBackend(verbose=False)
    t = SparseSignTransform(seed=0)
    try:
        t.basis(32, 8, backend, np.float32, A=None, B=None)
        assert False, "expected ValueError when A/B missing"
    except ValueError as e:
        assert "sparse_sign" in str(e)


def test_sparse_sign_rejects_bad_m():
    backend = CPUBackend(verbose=False)
    n = 32
    A = generate(n, np.float32, False, None, seed=1, fill="random")
    B = generate(n, np.float32, False, None, seed=2, fill="random")
    for bad_m in (0, -1, n + 1):
        try:
            SparseSignTransform(seed=0).basis(n, bad_m, backend, np.float32, A=A, B=B)
            assert False, f"expected ValueError for m={bad_m}"
        except ValueError as e:
            assert "sparse_sign" in str(e)


def test_sparse_sign_basis_shape_orthonormal():
    backend = CPUBackend(verbose=False)
    n, m = 64, 16
    A = generate(n, np.float32, False, None, seed=1, fill="lowrank", data_rank=4)
    B = generate(n, np.float32, False, None, seed=2, fill="lowrank", data_rank=4)
    Q = SparseSignTransform(seed=0).basis(n, m, backend, np.float32, A=A, B=B)
    Qh = backend.to_host(Q).astype(np.float64)

    assert Qh.shape == (n, m)
    assert np.isfinite(Qh).all()
    gram = Qh.T @ Qh
    assert float(np.linalg.norm(gram - np.eye(m))) < 1e-5


def test_sparse_sign_seed_deterministic():
    backend = CPUBackend(verbose=False)
    n, m = 48, 12
    A = generate(n, np.float32, False, None, seed=3, fill="random")
    B = generate(n, np.float32, False, None, seed=4, fill="random")
    Q1 = backend.to_host(SparseSignTransform(seed=7).basis(n, m, backend, np.float32, A, B))
    Q2 = backend.to_host(SparseSignTransform(seed=7).basis(n, m, backend, np.float32, A, B))
    Q3 = backend.to_host(SparseSignTransform(seed=8).basis(n, m, backend, np.float32, A, B))
    assert np.allclose(Q1, Q2)
    assert not np.allclose(Q1, Q3)


def test_sparse_sign_recovers_low_rank_product():
    # M comfortably exceeds 3x the rank: any reasonable 3-way basis should
    # recover the product almost exactly, the same easy-case check rsvd and
    # nystrom pass.
    backend = CPUBackend(verbose=False)
    n, rank, m = 128, 4, 60
    A = generate(n, np.float64, False, None, seed=10, fill="lowrank", data_rank=rank)
    B = generate(n, np.float64, False, None, seed=11, fill="lowrank", data_rank=rank)
    Q = backend.to_host(SparseSignTransform(seed=0).basis(n, m, backend, np.float64, A, B))
    P = Q @ Q.T
    C = A @ B
    Chat = P @ A @ P @ B @ P
    rel_err = np.linalg.norm(Chat - C) / np.linalg.norm(C)
    assert rel_err < 1e-8


def test_sparse_sign_basis_flops_honest():
    n, m = 12000, 1500
    ss = SparseSignTransform().basis_flops(n, m)
    ny = NystromTransform().basis_flops(n, m)
    rs = RandomizedSVDTransform().basis_flops(n, m)
    # Same QR term as nystrom (2*n*m^2), PLUS an honestly-counted term for the
    # (_SIGN_MIX - 1) real additions per output element that blending costs
    # and nystrom's single-column copy does not.
    expected = 2.0 * n * m * m + (SparseSignTransform._SIGN_MIX - 1) * n * m
    assert ss == expected
    assert ss > ny > 0.0          # strictly more than nystrom's pure-QR cost...
    assert ss < rs                # ...but still far cheaper than rsvd's O(n^2*m) sketch


def _rel_err(A, B, Q):
    P = Q @ Q.T
    C = A @ B
    Chat = P @ A @ P @ B @ P
    return np.linalg.norm(Chat - C) / np.linalg.norm(C)


def test_sparse_sign_beats_nystrom_on_decaying_spectrum():
    """Reproduces the pre-implementation numerical premise-check (see the
    transform's docstring and the PR) as an automated test: sparse_sign's
    blended-column sketch spans col(A)/row(A)/row(B) measurably better than
    nystrom's single-column draw on this project's own decaying-spectrum fill,
    using the real generate() and the real transform classes -- not a
    standalone numpy replica. Two regimes (a tough M==rank case and a
    moderate-headroom case), several fresh seeds each, comparing MEAN
    reconstruction accuracy (not a single lucky draw)."""
    backend = CPUBackend(verbose=False)
    for n, rank, m, trials in ((256, 64, 64, 6), (256, 32, 64, 6)):
        ny_accs, ss_accs = [], []
        for t in range(trials):
            seed = 500 + t
            A = generate(n, np.float64, False, None, seed=seed,
                         fill="decaying-spectrum", data_rank=rank)
            B = generate(n, np.float64, False, None, seed=seed + 1,
                         fill="decaying-spectrum", data_rank=rank)
            Qny = backend.to_host(NystromTransform(seed=seed).basis(n, m, backend, np.float64, A, B))
            Qss = backend.to_host(SparseSignTransform(seed=seed).basis(n, m, backend, np.float64, A, B))
            ny_accs.append(1.0 - _rel_err(A, B, Qny))
            ss_accs.append(1.0 - _rel_err(A, B, Qss))
        assert np.mean(ss_accs) > np.mean(ny_accs), (
            f"n={n} rank={rank} m={m}: sparse_sign mean acc {np.mean(ss_accs):.4f} "
            f"did not beat nystrom's {np.mean(ny_accs):.4f}"
        )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
