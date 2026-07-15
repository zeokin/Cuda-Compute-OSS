"""CPU-safe tests for the built-in ``power-rsvd`` transform.

No GPU required -- uses the smoke-test CPUBackend, so these run in the normal
contributor CI path (`pytest strategy/tests`). The GPU scorecard (accuracy,
latency, VRAM on the reference RTX 5090) still comes from `python -m eval`; these
tests prove the *core claim* -- power iteration lowers the reconstruction error
versus plain ``rsvd`` at the same M on a decaying spectrum -- on CPU, deterministically.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy.cpu_backend import CPUBackend
from strategy.storage import generate
from strategy.transforms import (
    PowerIterationTransform,
    RandomizedSVDTransform,
    available,
    get_transform,
)


def _subspace_rel_error(Q, A, B) -> float:
    """Relative Frobenius error of the subspace product Ĉ = Q(QᵀAQ)(QᵀBQ)Qᵀ vs A@B."""
    Q = Q.astype(np.float64)
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    Atil = Q.T @ A @ Q
    Btil = Q.T @ B @ Q
    Chat = Q @ (Atil @ Btil) @ Q.T
    C = A @ B
    return float(np.linalg.norm(Chat - C) / (np.linalg.norm(C) or 1.0))


def test_power_rsvd_registered():
    assert "power-rsvd" in available()
    assert isinstance(get_transform("power-rsvd"), PowerIterationTransform)


def test_power_rsvd_requires_operands():
    backend = CPUBackend(verbose=False)
    try:
        PowerIterationTransform(seed=0).basis(32, 8, backend, np.float32, A=None, B=None)
        assert False, "expected ValueError when A/B missing"
    except ValueError as e:
        assert "power-rsvd" in str(e)


def test_power_rsvd_rejects_bad_q():
    for bad in (-1, 1.5, True):
        try:
            PowerIterationTransform(seed=0, q=bad)
            assert False, f"expected ValueError for q={bad!r}"
        except ValueError:
            pass


def test_power_rsvd_basis_shape_orthonormal():
    backend = CPUBackend(verbose=False)
    n, m = 96, 24
    A = generate(n, np.float64, False, None, seed=1, fill="decaying-spectrum", data_rank=48)
    B = generate(n, np.float64, False, None, seed=2, fill="decaying-spectrum", data_rank=48)
    Q = PowerIterationTransform(seed=0, q=2).basis(n, m, backend, np.float64, A=A, B=B)
    Qh = backend.to_host(Q).astype(np.float64)
    assert Qh.shape == (n, m)
    assert np.isfinite(Qh).all()
    gram = Qh.T @ Qh
    assert float(np.linalg.norm(gram - np.eye(m))) < 1e-8


def test_power_rsvd_seed_deterministic():
    backend = CPUBackend(verbose=False)
    n, m = 64, 12
    A = generate(n, np.float64, False, None, seed=3, fill="decaying-spectrum", data_rank=32)
    B = generate(n, np.float64, False, None, seed=4, fill="decaying-spectrum", data_rank=32)
    Q1 = backend.to_host(PowerIterationTransform(seed=7, q=2).basis(n, m, backend, np.float64, A=A, B=B))
    Q2 = backend.to_host(PowerIterationTransform(seed=7, q=2).basis(n, m, backend, np.float64, A=A, B=B))
    Q3 = backend.to_host(PowerIterationTransform(seed=8, q=2).basis(n, m, backend, np.float64, A=A, B=B))
    assert np.allclose(Q1, Q2)
    assert not np.allclose(Q1, Q3)


def test_power_rsvd_q_zero_matches_rsvd():
    # q=0 is exactly the rsvd sketch (same seed, same widths) -> identical basis.
    backend = CPUBackend(verbose=False)
    n, m = 80, 18
    A = generate(n, np.float64, False, None, seed=5, fill="decaying-spectrum", data_rank=40)
    B = generate(n, np.float64, False, None, seed=6, fill="decaying-spectrum", data_rank=40)
    Q_power0 = backend.to_host(PowerIterationTransform(seed=0, q=0).basis(n, m, backend, np.float64, A=A, B=B))
    Q_rsvd = backend.to_host(RandomizedSVDTransform(seed=0).basis(n, m, backend, np.float64, A=A, B=B))
    assert np.allclose(Q_power0, Q_rsvd)


def test_power_rsvd_beats_rsvd_on_decaying_spectrum():
    # The core value claim: on a decaying (not exactly low-rank) spectrum, q
    # subspace-iteration steps align the basis with the dominant directions far
    # better than a single sketch, so the subspace reconstruction error drops --
    # at the SAME M and SAME seed. Averaged over a few couples for robustness.
    backend = CPUBackend(verbose=False)
    n, m, rank = 128, 30, 64
    rsvd_errs, power_errs = [], []
    for i in range(3):
        A = generate(n, np.float64, False, None, seed=10 + 2 * i, fill="decaying-spectrum",
                     data_rank=rank, spectral_alpha=0.5)
        B = generate(n, np.float64, False, None, seed=11 + 2 * i, fill="decaying-spectrum",
                     data_rank=rank, spectral_alpha=0.5)
        Q_rsvd = backend.to_host(RandomizedSVDTransform(seed=0).basis(n, m, backend, np.float64, A=A, B=B))
        Q_power = backend.to_host(PowerIterationTransform(seed=0, q=2).basis(n, m, backend, np.float64, A=A, B=B))
        rsvd_errs.append(_subspace_rel_error(Q_rsvd, A, B))
        power_errs.append(_subspace_rel_error(Q_power, A, B))
    rsvd_err = float(np.mean(rsvd_errs))
    power_err = float(np.mean(power_errs))
    # Strictly better, with a real margin (not just numerical noise).
    assert power_err < rsvd_err, f"power-rsvd {power_err:.4f} not < rsvd {rsvd_err:.4f}"
    assert power_err < 0.9 * rsvd_err, (
        f"expected power-rsvd to cut error >=10%: power={power_err:.4f} rsvd={rsvd_err:.4f}"
    )


def test_power_rsvd_basis_flops_reflect_q():
    # FLOPs must be reported honestly: strictly above rsvd, growing with q, so the
    # scorecard never overstates the savings.
    n, m = 8192, 1024
    rsvd = RandomizedSVDTransform(seed=0).basis_flops(n, m)
    p1 = PowerIterationTransform(seed=0, q=1).basis_flops(n, m)
    p2 = PowerIterationTransform(seed=0, q=2).basis_flops(n, m)
    assert PowerIterationTransform(seed=0, q=0).basis_flops(n, m) == rsvd
    assert p1 > rsvd
    assert p2 > p1
    # still O(N^2 M), far below the exact 2 N^3.
    assert p2 < 2.0 * n * n * n


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
