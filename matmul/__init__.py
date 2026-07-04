"""GPU matrix-multiplication system (square n x n, arbitrarily large n).

Quick API
---------
    import numpy as np
    from matmul import matmul

    A = np.random.rand(4096, 4096).astype("float32")
    B = np.random.rand(4096, 4096).astype("float32")
    C = matmul(A, B)            # GPU only (CUDA/MPS); raises if no GPU is present

For huge n that will not fit in RAM, use the disk-backed runner in
``matmul.runner`` or the CLI (``python -m matmul``).
"""
from __future__ import annotations

import numpy as np

from .config import Config, DTYPES
from .backend import Backend
from . import gemm, storage, runner

__all__ = ["Config", "Backend", "matmul", "DTYPES", "gemm", "storage", "runner"]


def matmul(A: np.ndarray, B: np.ndarray, out: np.ndarray | None = None,
           config: Config | None = None) -> np.ndarray:
    """Multiply two square in-memory matrices and return C = A @ B.

    A/B may be NumPy arrays or memmaps. If ``out`` is given it is written in
    place (and returned); otherwise a new array is allocated.
    """
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same size")
    cfg = config or Config(dtype=_dtype_name(A.dtype), verbose=False)
    backend = Backend(cfg.device, cfg.verbose)
    C = out if out is not None else np.empty_like(A, dtype=cfg.np_dtype)
    gemm.multiply(A, B, C, backend, cfg)
    return C


def _dtype_name(dt) -> str:
    dt = np.dtype(dt)
    for name, npdt in DTYPES.items():
        if np.dtype(npdt) == dt:
            return name
    return "fp32"
