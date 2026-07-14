"""Blocked matrix multiplication C = A @ B for square n x n matrices, on the GPU.

Two regimes, chosen automatically:

* in-core  : A, B, C all fit in device memory  -> one GPU GEMM (torch.bmm).
* out-of-core (tiled): stream T x T tiles from A/B (which may be disk-backed
  memmaps) to the GPU, accumulating C[i,j] = sum_k A[i,k] @ B[k,j].

The tiled path is what makes n = 128k / 256k possible on a single 24-32 GB GPU.
Every device-side product goes through the backend (``torch.bmm``).
"""
from __future__ import annotations

import math

import numpy as np

from .backend import Backend
from .config import Config
from .storage import bytes_human


def _tiles(n: int, T: int):
    """Yield (start, size) blocks covering range(n) in steps of T (ragged end)."""
    return [(s, min(T, n - s)) for s in range(0, n, T)]


def _tile_operand_bytes(cfg: Config) -> int:
    """Storage bytes per element for one operand tile resident on the device."""
    if cfg.np_dtype == np.float16 and cfg.accumulate_fp32:
        # _gemm_tiled_sync upcasts fp16 operand tiles to fp32 before bmm.
        return np.dtype(np.float32).itemsize
    return cfg.item_bytes


def _tile_workspace_bytes_per_elem(cfg: Config) -> int:
    """Device bytes per T×T element in one tiled (i, j, k) accumulation step.

    Four T×T tiles are steady-state live at each k-step -- the accumulator,
    the two operand tiles (A, B), and the freshly allocated GEMM output
    returned by ``backend.matmul`` (``torch.bmm`` cannot write its result into
    either operand) -- but ``_gemm_tiled_sync`` reassigns ``a_dev``, ``b_dev``,
    and ``prod`` every iteration (e.g. ``a_dev = backend.to_device(...)``), and
    Python evaluates the right-hand side of an assignment -- allocating the NEW
    tile -- before the name rebinds and the OLD tile's reference is dropped. So
    at the moment any one of those three lines runs, the previous iteration's
    tile it is about to replace is still live alongside the freshly allocated
    one: a fifth T×T tile, momentarily, on top of the steady-state four.
    Budgeting only four under-counts true peak device usage by one operand-
    sized tile, risking OOM at a ``vram_fraction`` that leaves little headroom.
    The output tile (and this fifth, transient one) are produced in the
    operand/compute dtype, so each costs one ``_tile_operand_bytes`` term.
    """
    return cfg.acc_dtype.itemsize + 4 * _tile_operand_bytes(cfg)


def auto_tile(n: int, cfg: Config, backend: Backend) -> int:
    """Pick tile edge T so the working set fits in the VRAM budget.

    Working set per (i,j,k) step on the device:
        acc (T x T, acc_dtype)
        + (A-tile + B-tile) (T x T, operand bytes)
        + prod (T x T, the bmm output, operand bytes)
    """
    budget = int(backend.free_compute_bytes() * cfg.vram_fraction)
    per_elem = _tile_workspace_bytes_per_elem(cfg)
    raw = int(math.sqrt(max(1, budget) / per_elem))
    t = min(max(1, raw), n)
    # Prefer 128-aligned GEMM shapes only when doing so stays within budget.
    if t >= 128:
        t = max(128, (t // 128) * 128)
    return t


def _in_core_bytes_per_elem(cfg: Config) -> int:
    """Peak device bytes per n×n element for the in-core GEMM path.

    _gemm_in_core keeps the A and B operands resident and, for the default
    fp16 + accumulate_fp32 path, also holds their fp32 upcasts plus the fp32
    product before the fp16 store — all live at once during the matmul:

        A, B (2 x operand)  +  A32, B32, out32 (3 x acc)

    A naive ``3 * item_bytes`` (A+B+C) under-budgets that fp16 case by ~2.7x
    (2*2 + 3*4 = 16 vs 3*2 = 6), letting a run pass the in-core check and then
    OOM mid-multiply instead of falling back to tiling.
    """
    if cfg.np_dtype == np.float16 and cfg.accumulate_fp32:
        return 2 * cfg.item_bytes + 3 * cfg.acc_dtype.itemsize
    return 3 * cfg.item_bytes


def _fits_in_core(n: int, cfg: Config, backend: Backend) -> bool:
    # Peak device residency for the in-core path (see _in_core_bytes_per_elem).
    need = n * n * _in_core_bytes_per_elem(cfg)
    return need <= backend.free_compute_bytes() * cfg.vram_fraction


# ---------------------------------------------------------------------------
# in-core
# ---------------------------------------------------------------------------
def _gemm_in_core(A, B, C, backend: Backend, cfg: Config) -> None:
    a = backend.to_device(np.asarray(A))
    b = backend.to_device(np.asarray(B))
    if cfg.np_dtype == np.float16 and cfg.accumulate_fp32:
        # Higher-accuracy path: accumulate in fp32, store fp16.
        c = backend.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    else:
        c = backend.matmul(a, b)
    C[...] = backend.to_host(c)
    if isinstance(C, np.memmap):
        C.flush()


# ---------------------------------------------------------------------------
# out-of-core, tiled
# ---------------------------------------------------------------------------
def _gemm_tiled_sync(A, B, C, backend: Backend, cfg: Config, T: int) -> None:
    xp = backend.xp
    n = A.shape[0]
    item = cfg.np_dtype
    acc_dtype = cfg.acc_dtype
    blocks = _tiles(n, T)

    host_budget = int(backend.host_available_bytes() * 0.4)

    for (r0, ti) in blocks:
        # Cache the A row-panel in host RAM once per i, if it fits, to avoid
        # re-reading it from disk for every column block j.
        panel_fits = ti * n * item.itemsize <= host_budget
        A_panel = np.ascontiguousarray(A[r0 : r0 + ti, :]) if panel_fits else None

        for (c0, tj) in blocks:
            acc = xp.zeros((ti, tj), dtype=acc_dtype)
            for (k0, tk) in blocks:
                if panel_fits:
                    a_host = A_panel[:, k0 : k0 + tk]
                else:
                    a_host = A[r0 : r0 + ti, k0 : k0 + tk]
                b_host = B[k0 : k0 + tk, c0 : c0 + tj]

                a_dev = backend.to_device(a_host)
                b_dev = backend.to_device(b_host)
                if cfg.np_dtype == np.float16 and cfg.accumulate_fp32:
                    # Mirror _gemm_in_core: accumulate tile products in fp32.
                    a_dev = a_dev.astype(np.float32)
                    b_dev = b_dev.astype(np.float32)
                prod = backend.matmul(a_dev, b_dev)
                if prod.dtype != acc_dtype:
                    prod = prod.astype(acc_dtype)
                acc += prod

            out = acc if acc_dtype == item else acc.astype(item)
            C[r0 : r0 + ti, c0 : c0 + tj] = backend.to_host(out)

    if isinstance(C, np.memmap):
        C.flush()


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def multiply(A, B, C, backend: Backend, cfg: Config) -> dict:
    """Compute C = A @ B. Returns a dict describing what was done."""
    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n) or C.shape != (n, n):
        raise ValueError("A, B, C must all be square n x n with matching n")

    # Disk-backed memmaps must use the tiled path: np.asarray() in _gemm_in_core
    # would materialise the full matrix in host RAM, defeating out-of-core storage.
    on_disk = any(isinstance(x, np.memmap) for x in (A, B, C))
    if _fits_in_core(n, cfg, backend) and not cfg.force_tiled and not on_disk:
        mode, T = "in-core", n
        _gemm_in_core(A, B, C, backend, cfg)
    else:
        T = cfg.tile or auto_tile(n, cfg, backend)
        mode = f"tiled-sync(T={T})"
        _gemm_tiled_sync(A, B, C, backend, cfg, T)

    return {
        "n": n,
        "mode": mode,
        "tile": T,
        "device": backend.name,
        "dtype": cfg.dtype,
        "working_set": bytes_human(3 * n * n * cfg.item_bytes),
    }
