"""CPU-safe tests for the built-in ``nystrom`` transform.

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
    RandomizedSVDTransform,
    available,
    get_transform,
)


def test_nystrom_registered():
    assert "nystrom" in available()
    assert isinstance(get_transform("nystrom"), NystromTransform)


def test_nystrom_requires_operands():
    backend = CPUBackend(verbose=False)
    t = NystromTransform(seed=0)
    try:
        t.basis(32, 8, backend, np.float32, A=None, B=None)
        assert False, "expected ValueError when A/B missing"
    except ValueError as e:
        assert "nystrom" in str(e)


def test_nystrom_basis_shape_orthonormal():
    backend = CPUBackend(verbose=False)
    n, m = 64, 16
    A = generate(n, np.float32, False, None, seed=1, fill="lowrank", data_rank=4)
    B = generate(n, np.float32, False, None, seed=2, fill="lowrank", data_rank=4)
    Q = NystromTransform(seed=0).basis(n, m, backend, np.float32, A=A, B=B)
    Qh = backend.to_host(Q).astype(np.float64)

    assert Qh.shape == (n, m)
    assert np.isfinite(Qh).all()
    gram = Qh.T @ Qh
    assert float(np.linalg.norm(gram - np.eye(m))) < 1e-5


def test_nystrom_seed_deterministic():
    backend = CPUBackend(verbose=False)
    n, m = 48, 12
    A = generate(n, np.float32, False, None, seed=3, fill="random")
    B = generate(n, np.float32, False, None, seed=4, fill="random")
    Q1 = backend.to_host(NystromTransform(seed=7).basis(n, m, backend, np.float32, A, B))
    Q2 = backend.to_host(NystromTransform(seed=7).basis(n, m, backend, np.float32, A, B))
    Q3 = backend.to_host(NystromTransform(seed=8).basis(n, m, backend, np.float32, A, B))
    assert np.allclose(Q1, Q2)
    assert not np.allclose(Q1, Q3)


def test_nystrom_basis_flops_honest_and_cheaper_than_rsvd():
    n, m = 12000, 1500
    ny = NystromTransform().basis_flops(n, m)
    assert ny == 2.0 * n * m * m
    assert ny > 0.0
    assert ny < RandomizedSVDTransform().basis_flops(n, m)
