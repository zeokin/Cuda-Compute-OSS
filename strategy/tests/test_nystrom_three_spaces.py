"""nystrom splits M across 3 spaces (drop redundant col(B)) (#269)."""
from __future__ import annotations

import numpy as np

from strategy.backend import NumpyBackend
from strategy.transforms import NystromTransform


def test_nystrom_recovers_rank4_at_m15():
    """4-space split fails at M=15 (3/space < rank); 3-space recovers."""
    rng = np.random.default_rng(0)
    n, r, m = 64, 4, 15
    X = rng.standard_normal((n, r)).astype(np.float64)
    Y = rng.standard_normal((n, r)).astype(np.float64)
    A = (X @ Y.T).astype(np.float32)
    B = (Y @ X.T).astype(np.float32)  # keep product low-rank-ish
    # Simpler: A=UU^T style low-rank factors
    U = rng.standard_normal((n, r)).astype(np.float32)
    V = rng.standard_normal((n, r)).astype(np.float32)
    A = U @ V.T
    B = V @ U.T
    backend = NumpyBackend()
    Q = NystromTransform(seed=0).basis(n, m, backend, np.float32, A=A, B=B)
    Qh = backend.to_host(Q)
    P = Qh @ Qh.T
    approx = P @ A @ P @ B @ P
    exact = A @ B
    rel = np.linalg.norm(exact - approx, "fro") / max(np.linalg.norm(exact, "fro"), 1e-12)
    assert rel < 1e-5, rel
