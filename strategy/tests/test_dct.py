"""CPU-safe tests for the built-in ``dct`` transform.

No GPU required — uses the smoke-test CPUBackend so these always run in the
contributor CI path (`pytest strategy/tests`).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy.cpu_backend import CPUBackend
from strategy.transforms import DCTTransform, available, get_transform


def test_dct_registered():
    assert "dct" in available()
    assert isinstance(get_transform("dct"), DCTTransform)


def test_dct_basis_shape_orthonormal_no_operands():
    # Fixed basis: A/B are optional and unused.
    backend = CPUBackend(verbose=False)
    n, m = 64, 16
    Q = DCTTransform(seed=0).basis(n, m, backend, np.float32, A=None, B=None)
    Qh = backend.to_host(Q).astype(np.float64)

    assert Qh.shape == (n, m)
    assert np.isfinite(Qh).all()
    gram = Qh.T @ Qh
    assert float(np.linalg.norm(gram - np.eye(m))) < 1e-6


def test_dct_basis_rejects_invalid_m():
    backend = CPUBackend(verbose=False)
    t = DCTTransform()
    try:
        t.basis(8, 0, backend, np.float32)
        assert False, "expected ValueError for m=0"
    except ValueError:
        pass
    try:
        t.basis(8, 9, backend, np.float32)
        assert False, "expected ValueError for m>n"
    except ValueError:
        pass


def test_dct_basis_flops_honest():
    n, m = 12000, 1500
    assert DCTTransform().basis_flops(n, m) == float(n * m)
    assert DCTTransform().basis_flops(n, m) > 0.0


def test_dct_projector_exact_on_dct_subspace():
    # A vector in the span of the first m DCT modes is recovered exactly by
    # the orthogonal projector Q Qᵀ.
    backend = CPUBackend(verbose=False)
    n, m = 48, 12
    Q = backend.to_host(
        DCTTransform().basis(n, m, backend, np.float64)
    ).astype(np.float64)
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(m)
    v = Q @ coeffs
    recovered = Q @ (Q.T @ v)
    assert float(np.linalg.norm(recovered - v)) < 1e-9
