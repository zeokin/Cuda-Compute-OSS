"""The subspace ('smart') strategy: compress -> compute -> reconstruct.

    (N,N) --compress-->  Atil = Q^T A Q   (M,M)
                         Btil = Q^T B Q   (M,M)
           --compute-->  Ctil = Atil @ Btil          (M,M, cheap)
           --reconstruct-> C  = Q Ctil Q^T           (N,N)

with an orthonormal N x M basis Q from a pluggable transform. This equals
P A P B P with projector P = Q Q^T, so it is EXACT only when M = N or when A/B
live in the subspace Q captures (low rank / smooth). Otherwise it is an
approximation whose quality is set entirely by the transform.

Cost: O(N^2 M) vs O(N^3) for the exact product (FLOP ratio ~ 4M/N once the basis
construction is counted, not ~3M/N). The streaming helpers read the big matrices
one row-block at a time, so A/B may be disk-backed memmaps far larger than RAM/VRAM.

Standalone: no imports from the sibling `matmul` package.
"""
from __future__ import annotations

import numpy as np

from .backend import Backend
from .config import Config
from .storage import bytes_human
from .transforms import get_transform


# Fraction of free device memory a single streamed row-block may occupy when no
# explicit budget is supplied (e.g. the primitives are called directly). Real
# strategy runs pass ``cfg.vram_fraction`` through instead.
_DEFAULT_ROW_BLOCK_FRACTION = 0.3


def _row_block(n: int, cols: int, backend: Backend, item_bytes: int,
               frac: float = _DEFAULT_ROW_BLOCK_FRACTION) -> int:
    """Choose how many rows of an (n x cols) stream to stage on the device.

    ``frac`` is the fraction of free device memory one row-block may use
    (``Config.vram_fraction`` when driven by the strategy)."""
    budget = int(backend.free_compute_bytes() * frac)
    per_row = max(1, cols * item_bytes)
    return int(min(n, max(1, budget // per_row)))


# ---------------------------------------------------------------------------
# streaming BLAS-3 primitives (row-block streamed; memmap-friendly)
# ---------------------------------------------------------------------------
def stream_gemm_right(X, Q, backend: Backend, dtype,
                      frac: float = _DEFAULT_ROW_BLOCK_FRACTION):
    """Return X @ Q  (n x m), streaming the rows of X. Q is resident (n x m)."""
    xp = backend.xp
    n, m = X.shape[0], Q.shape[1]
    out = xp.empty((n, m), dtype=dtype)
    blk = _row_block(n, X.shape[1], backend, np.dtype(dtype).itemsize, frac)
    for r0 in range(0, n, blk):
        r1 = min(n, r0 + blk)
        Xr = backend.to_device(np.asarray(X[r0:r1, :]).astype(dtype, copy=False))
        out[r0:r1, :] = backend.matmul(Xr, Q)
    return out


def stream_gemm_left_t(X, Q, backend: Backend, dtype,
                       frac: float = _DEFAULT_ROW_BLOCK_FRACTION):
    """Return X^T @ Q  (n x m) for square X, streaming the rows of X:
    X^T @ Q = sum over row-blocks of X[rb,:]^T @ Q[rb,:]."""
    xp = backend.xp
    n, m = X.shape[0], Q.shape[1]
    acc = xp.zeros((n, m), dtype=dtype)
    blk = _row_block(n, X.shape[1], backend, np.dtype(dtype).itemsize, frac)
    for r0 in range(0, n, blk):
        r1 = min(n, r0 + blk)
        Xr = backend.to_device(np.asarray(X[r0:r1, :]).astype(dtype, copy=False))
        acc += backend.matmul(Xr.T, Q[r0:r1, :])
    return acc


def compress(X, Q, backend: Backend, dtype,
             frac: float = _DEFAULT_ROW_BLOCK_FRACTION):
    """Return Q^T X Q  (m x m), streaming the rows of X:
    Q^T X Q = sum over row-blocks  Q[rb,:]^T @ (X[rb,:] @ Q)."""
    xp = backend.xp
    n, m = X.shape[0], Q.shape[1]
    acc = xp.zeros((m, m), dtype=dtype)
    blk = _row_block(n, X.shape[1], backend, np.dtype(dtype).itemsize, frac)
    for r0 in range(0, n, blk):
        r1 = min(n, r0 + blk)
        Xr = backend.to_device(np.asarray(X[r0:r1, :]).astype(dtype, copy=False))
        acc += backend.matmul(Q[r0:r1, :].T, backend.matmul(Xr, Q))
    return acc


def reconstruct(Ctil, Q, C_out, backend: Backend, out_dtype,
                frac: float = _DEFAULT_ROW_BLOCK_FRACTION):
    """Write Q @ Ctil @ Q^T  (n x n) into C_out, streaming output row-blocks."""
    n = Q.shape[0]
    blk = _row_block(n, n, backend, np.dtype(out_dtype).itemsize, frac)
    QT = Q.T
    for r0 in range(0, n, blk):
        r1 = min(n, r0 + blk)
        outr = backend.matmul(backend.matmul(Q[r0:r1, :], Ctil), QT)   # (rb, n)
        C_out[r0:r1, :] = backend.to_host(outr).astype(out_dtype, copy=False)
    if isinstance(C_out, np.memmap):
        C_out.flush()


# ---------------------------------------------------------------------------
# smart (subspace) multiply
# ---------------------------------------------------------------------------
def _default_rank(n: int) -> int:
    return int(min(n, max(64, n // 8)))


def _flop_actual(n: int, m: int) -> float:
    """FLOPs for the CORE stages of one multiply_subspace call (basis excluded --
    ``multiply_subspace`` adds ``transform.basis_flops(n, m)`` on top, since basis
    cost is transform-specific).

    Two ``compress()`` calls (A and B), each doing ``X @ Q`` (2n^2m) then
    ``Q.T @ (X @ Q)`` (2nm^2); the (m,m) core product Atil @ Btil (2m^3); and
    ``reconstruct()``, doing ``Q @ Ctil`` (2nm^2) then ``(...) @ Q.T`` (2n^2m).
    """
    compress = 2 * n * n * m + 2 * n * m * m
    core = 2.0 * m * m * m
    reconstruct = 2 * n * m * m + 2 * n * n * m
    return 2.0 * compress + core + reconstruct


def multiply_subspace(A, B, C, backend: Backend, cfg: Config) -> dict:
    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n) or C.shape != (n, n):
        raise ValueError("A, B, C must all be square n x n with matching n")
    m = cfg.rank_m or _default_rank(n)
    if not (1 <= m <= n):
        raise ValueError(f"rank_m must be in [1, n]; got {m} for n={n}")
    cdt = cfg.compute_dtype

    transform = get_transform(cfg.transform, cfg.transform_seed)
    Q = transform.basis(n, m, backend, cdt, A=A, B=B)     # (n, m) orthonormal
    if Q.shape != (n, m):
        raise ValueError(f"transform returned basis {Q.shape}, expected {(n, m)}")

    frac = cfg.vram_fraction
    Atil = compress(A, Q, backend, cdt, frac)             # (m, m)
    Btil = compress(B, Q, backend, cdt, frac)             # (m, m)
    Ctil = backend.matmul(Atil, Btil)                     # (m, m)  -- cheap core
    reconstruct(Ctil, Q, C, backend, cfg.np_dtype, frac)

    return {
        "n": n,
        "strategy": "subspace",
        "mode": f"subspace(M={m}, transform={transform.name})",
        "rank_m": m,
        "transform": transform.name,
        "device": backend.name,
        "dtype": cfg.dtype,
        "working_set": bytes_human(3 * n * n * cfg.item_bytes),
        "flop_exact": 2.0 * n * n * n,
        # core stages PLUS the transform's basis construction -- a mandatory
        # O(N^2 M) per-call cost (e.g. rsvd's sketches + QR) that would otherwise
        # be omitted, overstating the reported savings.
        "flop_actual": _flop_actual(n, m) + transform.basis_flops(n, m),
    }


# ---------------------------------------------------------------------------
# exact baseline (self-contained; for the normal-vs-smart comparison)
# ---------------------------------------------------------------------------
def multiply_exact(A, B, C, backend: Backend, cfg: Config) -> dict:
    """Full C = A @ B baseline. B is resident on the device; the rows of A are
    streamed. Fine for comparison-scale n; for out-of-core exact multiply at
    huge n use the sibling `matmul` package."""
    n = A.shape[0]
    dt = cfg.compute_dtype
    Bdev = backend.to_device(np.asarray(B).astype(dt, copy=False))
    blk = _row_block(n, n, backend, np.dtype(dt).itemsize, cfg.vram_fraction)
    for r0 in range(0, n, blk):
        r1 = min(n, r0 + blk)
        Ar = backend.to_device(np.asarray(A[r0:r1, :]).astype(dt, copy=False))
        Cr = backend.matmul(Ar, Bdev)
        C[r0:r1, :] = backend.to_host(Cr).astype(cfg.np_dtype, copy=False)
    if isinstance(C, np.memmap):
        C.flush()
    return {
        "n": n,
        "strategy": "exact",
        "mode": "exact(streamed)",
        "device": backend.name,
        "dtype": cfg.dtype,
        "flop_exact": 2.0 * n * n * n,
        "flop_actual": 2.0 * n * n * n,
    }
