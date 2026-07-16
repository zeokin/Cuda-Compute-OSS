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

import inspect
import math

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
               frac: float = _DEFAULT_ROW_BLOCK_FRACTION,
               out_cols: int = 0, fixed_bytes: int = 0) -> int:
    """Choose how many rows of an (n x cols) stream to stage on the device.

    ``frac`` is the fraction of free device memory one row-block may use
    (``Config.vram_fraction`` when driven by the strategy).

    A block costs more than the rows it stages: each iteration also allocates the
    GEMM output, which cannot alias its operands and is live alongside the staged
    block. ``out_cols`` is the number of output columns produced *per staged row*
    (so the real per-row cost is ``(cols + out_cols) * item_bytes``);
    ``fixed_bytes`` is any per-iteration allocation whose size does not scale with
    the block (e.g. ``stream_gemm_left_t``'s full ``(n, m)`` product), and is taken
    off the budget up front. Counting only ``cols`` under-budgets the block by up
    to 2x at ``M = N`` (see #138, and #95 for the same fix in ``matmul/gemm.py``).
    """
    budget = int(backend.free_compute_bytes() * frac) - int(fixed_bytes)
    per_row = max(1, (cols + out_cols) * item_bytes)
    return int(min(n, max(1, budget // per_row)))


def _exact_tile(n: int, backend: Backend, item_bytes: int, frac: float) -> int:
    """Tile edge for row- and k-blocking in ``multiply_exact``.

    Per (row, k) step the device holds an accumulator (T x n), an A panel
    (T x T), a B panel (T x n), and the GEMM output ``matmul(Ar, Bk)`` (T x n) --
    which cannot alias either operand and is live while it is folded into ``acc``.
    That is T * (3n + T) elements, so solve T^2 + 3nT - budget = 0 (see #144;
    #95 and #138 are the same omission in ``matmul/gemm.py`` and ``_row_block``).
    """
    budget_elems = max(1, int(backend.free_compute_bytes() * frac) // item_bytes)
    t = int((math.sqrt(9 * n * n + 4 * budget_elems) - 3 * n) / 2)
    return max(1, min(t, n))


# ---------------------------------------------------------------------------
# streaming BLAS-3 primitives (row-block streamed; memmap-friendly)
# ---------------------------------------------------------------------------
def stream_gemm_right(X, Q, backend: Backend, dtype,
                      frac: float = _DEFAULT_ROW_BLOCK_FRACTION,
                      extra_fixed_bytes: int = 0):
    """Return X @ Q  (n x m), streaming the rows of X. Q is resident (n x m).

    ``extra_fixed_bytes`` is caller-resident device memory that stays live for
    the whole call (e.g. earlier rsvd sketch parts) and is invisible to
    ``free_compute_bytes()`` on MPS, so it must be charged like ``out``.
    """
    xp = backend.xp
    n, m = X.shape[0], Q.shape[1]
    item = np.dtype(dtype).itemsize
    out = xp.empty((n, m), dtype=dtype)
    # `out` is a full (n, m) device buffer resident for the whole loop, so it is a
    # fixed cost that does not scale with the block -- charge it up front, exactly
    # as stream_gemm_left_t does for its (n, m) accumulator. Omitting it sizes the
    # block against the whole budget and under-counts device use by n*m (up to the
    # entire budget at M = N), risking OOM. Each block also allocates
    # matmul(Xr, Q) -> (blk, m): m output cols per staged row.
    blk = _row_block(n, X.shape[1], backend, item, frac,
                     out_cols=m,
                     fixed_bytes=n * m * item + int(extra_fixed_bytes))
    for r0 in range(0, n, blk):
        r1 = min(n, r0 + blk)
        Xr = backend.to_device(np.asarray(X[r0:r1, :]).astype(dtype, copy=False))
        out[r0:r1, :] = backend.matmul(Xr, Q)
    return out


def stream_gemm_left_t(X, Q, backend: Backend, dtype,
                       frac: float = _DEFAULT_ROW_BLOCK_FRACTION,
                       extra_fixed_bytes: int = 0):
    """Return X^T @ Q  (n x m) for square X, streaming the rows of X:
    X^T @ Q = sum over row-blocks of X[rb,:]^T @ Q[rb,:].

    ``extra_fixed_bytes`` is caller-resident device memory that stays live for
    the whole call (e.g. earlier rsvd sketch parts) and is invisible to
    ``free_compute_bytes()`` on MPS, so it must be charged like ``acc``.
    """
    xp = backend.xp
    n, m = X.shape[0], Q.shape[1]
    acc = xp.zeros((n, m), dtype=dtype)
    # Two (n, m) buffers are live at the peak of each step and neither scales with
    # the block: the resident accumulator ``acc`` and the ``matmul(Xr.T, Q[rb])``
    # product, which cannot alias ``acc`` and coexists with it during ``acc += ...``.
    # Charge both up front (2*n*m) -- counting only the product (n*m) leaves the
    # accumulator unbudgeted and sizes the block against the whole budget, risking
    # OOM (cf. #138 and stream_gemm_right's resident output).
    item = np.dtype(dtype).itemsize
    blk = _row_block(n, X.shape[1], backend, item, frac,
                     fixed_bytes=2 * n * m * item + int(extra_fixed_bytes))
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
    # Two (m, m) buffers are live and neither scales with the block: the resident
    # accumulator ``acc`` and the ``matmul(...) -> (m, m)`` product, which cannot
    # alias ``acc`` and coexists with it during ``acc += ...``. Charge both up
    # front (2*m*m) -- counting only the product leaves the accumulator
    # unbudgeted (invisible on MPS, where free_compute_bytes() is a static
    # ceiling), exactly as stream_gemm_left_t does for its (n, m) accumulator.
    # Each block also stages Xr and its matmul(Xr, Q) -> (blk, m) intermediate.
    item = np.dtype(dtype).itemsize
    blk = _row_block(n, X.shape[1], backend, item, frac,
                     out_cols=m, fixed_bytes=2 * m * m * item)
    for r0 in range(0, n, blk):
        r1 = min(n, r0 + blk)
        Xr = backend.to_device(np.asarray(X[r0:r1, :]).astype(dtype, copy=False))
        acc += backend.matmul(Q[r0:r1, :].T, backend.matmul(Xr, Q))
    return acc


def reconstruct(Ctil, Q, C_out, backend: Backend, out_dtype,
                frac: float = _DEFAULT_ROW_BLOCK_FRACTION, compute_dtype=None):
    """Write Q @ Ctil @ Q^T  (n x n) into C_out, streaming output row-blocks.

    The per-row device footprint is sized from ``compute_dtype`` — the dtype the
    loop tensors (``Q``, ``Ctil``, ``QT`` and the ``(rb, n)`` product) actually
    live in — not ``out_dtype``. For ``fp16`` inputs those differ (compute is
    bumped to fp32 for accuracy), so sizing from ``out_dtype`` would under-budget
    the row-block by 2x and risk OOM on large-N / tight-``vram_fraction`` runs.
    ``compute_dtype=None`` falls back to ``out_dtype`` (they match unless bumped).
    """
    n, m = Q.shape
    item_dtype = compute_dtype if compute_dtype is not None else out_dtype
    item = np.dtype(item_dtype).itemsize
    # per row: the (rb, n) product plus the (rb, m) intermediate Q[rb] @ Ctil,
    # which is still live while the outer matmul against QT runs. Q (n, m) and
    # Ctil (m, m) are also both fully resident for the entire loop -- unlike
    # the streamed inputs elsewhere in this module, they arrive already on the
    # device and are never staged per block. That is a fixed n*m + m*m cost
    # that does not shrink with the block, so it must be taken off the budget
    # up front (cf. stream_gemm_right's resident output, stream_gemm_left_t's
    # resident accumulator) rather than left to a live free-memory reading
    # that, on MPS, is a static ceiling and never reflects it at all.
    blk = _row_block(n, n, backend, item, frac,
                     out_cols=m, fixed_bytes=(n * m + m * m) * item)
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
def default_rank_m(n: int) -> int:
    """Default subspace dimension M when ``rank_m`` is unset.

    Matches the strategy's real default: ``min(n, max(64, n // 8))``. The floor
    at 64 keeps tiny-N smoke runs numerically stable; eval must report this same
    value so scorecards reproduce what was actually multiplied.
    """
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
    m = cfg.rank_m if cfg.rank_m is not None else default_rank_m(n)
    if not (1 <= m <= n):
        raise ValueError(f"rank_m must be in [1, n]; got {m} for n={n}")
    cdt = cfg.compute_dtype

    frac = cfg.vram_fraction
    transform = get_transform(cfg.transform, cfg.transform_seed)
    # Give the basis stage the same VRAM budget as compress/reconstruct. Pass frac
    # only if the transform accepts it, so custom transforms with the older
    # basis(...) signature keep working.
    if "frac" in inspect.signature(transform.basis).parameters:
        Q = transform.basis(n, m, backend, cdt, A=A, B=B, frac=frac)  # (n, m) orthonormal
    else:
        Q = transform.basis(n, m, backend, cdt, A=A, B=B)
    if Q.shape != (n, m):
        raise ValueError(f"transform returned basis {Q.shape}, expected {(n, m)}")
    Atil = compress(A, Q, backend, cdt, frac)             # (m, m)
    Btil = compress(B, Q, backend, cdt, frac)             # (m, m)
    Ctil = backend.matmul(Atil, Btil)                     # (m, m)  -- cheap core
    reconstruct(Ctil, Q, C, backend, cfg.np_dtype, frac, cdt)

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
    """Full C = A @ B baseline. Both A and B are streamed in row/k blocks so
    disk-backed memmaps never fully materialise in host or device RAM.
    For huge-n exact multiply at maximum throughput use the sibling `matmul`
    package."""
    n = A.shape[0]
    dt = cfg.compute_dtype
    item = np.dtype(dt).itemsize
    T = _exact_tile(n, backend, item, cfg.vram_fraction)
    xp = backend.xp
    for r0 in range(0, n, T):
        r1 = min(n, r0 + T)
        ti = r1 - r0
        acc = xp.zeros((ti, n), dtype=dt)
        for k0 in range(0, n, T):
            k1 = min(n, k0 + T)
            Ar = backend.to_device(
                np.asarray(A[r0:r1, k0:k1]).astype(dt, copy=False)
            )
            Bk = backend.to_device(
                np.asarray(B[k0:k1, :]).astype(dt, copy=False)
            )
            # In-place: `acc = acc + prod` would build a second (ti, n) tensor
            # while the old acc is still live, on top of the product itself.
            acc += backend.matmul(Ar, Bk)
        C[r0:r1, :] = backend.to_host(acc).astype(cfg.np_dtype, copy=False)
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
