"""Subspace ('smart') matrix-multiplication strategy.

    (N,N) --compress--> (M,M) --compute--> (M,M) --reconstruct--> (N,N)

Approximate multiply for compressible (low-rank / smooth) matrices at O(N^2 M)
instead of O(N^3). The subspace basis Q is supplied by a pluggable, updatable
transform (transforms.py). Standalone package -- it does not import from the
sibling `matmul` (exact) package.

Quick API
---------
    import numpy as np
    from strategy import subspace_matmul, Config

    A = ...  # (n, n), ideally low-rank / smooth
    B = ...
    C = subspace_matmul(A, B, config=Config(transform="rsvd", rank_m=256))

Remember: this is APPROXIMATE. On full-rank random data with M << N the error
is ~100%. Always check the reconstruction error for your data.
"""
from __future__ import annotations

import numpy as np

from .config import Config, DTYPES
from .backend import Backend
from . import subspace, transforms, storage, runner
from .transforms import Transform, register_transform, available

__all__ = [
    "Config", "Backend", "DTYPES", "subspace_matmul",
    "subspace", "transforms", "storage", "runner",
    "Transform", "register_transform", "available",
]


def subspace_matmul(A: np.ndarray, B: np.ndarray, out: np.ndarray | None = None,
                    config: Config | None = None) -> np.ndarray:
    """Approximate C = A @ B via compress -> compute -> reconstruct.
    Returns C (newly allocated unless ``out`` is given)."""
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same size")
    cfg = config or Config(dtype=_dtype_name(A.dtype), verbose=False)
    backend = Backend(cfg.device, cfg.verbose)
    C = out if out is not None else np.empty_like(A, dtype=cfg.np_dtype)
    subspace.multiply_subspace(A, B, C, backend, cfg)
    return C


def _dtype_name(dt) -> str:
    dt = np.dtype(dt)
    for name, npdt in DTYPES.items():
        if np.dtype(npdt) == dt:
            return name
    return "fp32"
