"""Matrix storage: in-RAM ndarrays or on-disk memory-mapped files.

A memmap lets us address a 65 GB (128k x 128k fp32) matrix without ever
holding it fully in RAM; slicing a tile pulls only that tile off disk.
"""
from __future__ import annotations

import math
import os
from numbers import Integral, Real

import numpy as np


def _fill_random(mat: np.ndarray, seed: int, scale: float = 1.0) -> None:
    """Fill a (possibly memmapped) n x n matrix with N(0, scale^2) values,
    one row-block at a time so we never materialise the whole thing in RAM.

    ``chunk`` is scaled in place: ``standard_normal`` already returns float64, so
    an out-of-place ``* scale`` keeps the raw draw alive alongside the scaled
    copy and costs TWO full row-blocks, doubling the ~256 MiB this loop budgets
    for one. Same fix, same reason as ``_fill_lowrank``'s V (#297); ``*=`` is
    bit-identical to ``*``, and this is the DEFAULT fill for both the CLI and
    ``python -m eval``.
    """
    n = mat.shape[0]
    rng = np.random.default_rng(seed)
    # Keep each generated block near ~256 MiB of fp64 temporaries.
    block = max(1, min(n, (256 * 1024**2) // (n * 8)))
    for r0 in range(0, n, block):
        r1 = min(n, r0 + block)
        chunk = rng.standard_normal((r1 - r0, n))
        chunk *= scale
        mat[r0:r1, :] = chunk.astype(mat.dtype, copy=False)
    if isinstance(mat, np.memmap):
        mat.flush()


def _fill_iota(mat: np.ndarray, seed: int = 0) -> None:
    """Cheap seeded fill for fast benchmarking:
    value = (i + j + row_shift[i] + col_shift[j]) mod 97.

    The base ``i + j`` keeps the deterministic iota band; ``row_shift`` and
    ``col_shift`` are drawn from ``seed``. Previously ``seed`` entered only as a
    single additive constant ``(i + j + seed) % 97``, so distinct seeds produced
    the *same* matrix up to a global shift: a couple's A and B (seeds s, s+1) and
    successive couples were trivial offsets of one another, and eval effectively
    benchmarked ``A @ A`` on one repeated input (issue #104). Per-row/column
    shifts make distinct seeds statistically independent while staying O(n) cheap
    and deterministic per seed.

    The modulo is taken in place for the same reason ``_fill_random`` scales in
    place: ``(rows + cols) % 97`` keeps the int64 sum alive alongside its
    remainder, so one row-block costs two, doubling the ~256 MiB budgeted below.
    ``%=`` gives the identical result."""
    n = mat.shape[0]
    rng = np.random.default_rng(seed)
    row_shift = rng.integers(0, 97, size=n)
    col_shift = rng.integers(0, 97, size=n)
    block = max(1, min(n, (256 * 1024**2) // (n * 8)))
    cols = np.arange(n) + col_shift
    for r0 in range(0, n, block):
        r1 = min(n, r0 + block)
        rows = (np.arange(r0, r1) + row_shift[r0:r1])[:, None]
        values = rows + cols
        values %= 97
        mat[r0:r1, :] = values.astype(mat.dtype, copy=False)
    if isinstance(mat, np.memmap):
        mat.flush()


def _fill_lowrank(mat: np.ndarray, seed: int, rank: int) -> None:
    """Fill with a rank-``rank`` matrix (U @ V, rank << n) row-block at a time.
    This is the regime where the subspace strategy is accurate.

    ``V`` is built in place: ``standard_normal`` already returns float64, so an
    out-of-place ``* scale`` plus a redundant ``.astype(np.float64)`` would keep
    two full ``(rank, n)`` buffers alive (~2× host peak) even when ``mat`` is a
    disk memmap — enough to OOM generation at large n / default rank.
    """
    n = mat.shape[0]
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(rank)
    V = rng.standard_normal((rank, n))
    V *= scale
    block = max(1, min(n, (256 * 1024**2) // (n * 8)))
    for r0 in range(0, n, block):
        r1 = min(n, r0 + block)
        U = rng.standard_normal((r1 - r0, rank))
        mat[r0:r1, :] = (U @ V).astype(mat.dtype, copy=False)
    if isinstance(mat, np.memmap):
        mat.flush()


def _fill_decaying_spectrum(mat: np.ndarray, seed: int, rank: int, alpha: float = 1.0) -> None:
    """Fill with a rank-``rank`` matrix whose component weights decay as
    k^-alpha (k=1..rank), unlike _fill_lowrank's uniform weighting. Most of
    the energy sits in the first few components with a long, genuinely small
    (not zero) tail -- tests whether a transform prioritizes the strongest
    structure rather than needing the full rank to be accurate.

    Same in-place ``V`` construction as ``_fill_lowrank`` — avoid a second
    full-size float64 temporary during scale (and the useless same-dtype copy).
    """
    n = mat.shape[0]
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(rank)
    V = rng.standard_normal((rank, n))
    V *= scale
    weights = np.arange(1, rank + 1, dtype=np.float64) ** -alpha
    V *= weights[:, None]
    block = max(1, min(n, (256 * 1024**2) // (n * 8)))
    for r0 in range(0, n, block):
        r1 = min(n, r0 + block)
        U = rng.standard_normal((r1 - r0, rank))
        mat[r0:r1, :] = (U @ V).astype(mat.dtype, copy=False)
    if isinstance(mat, np.memmap):
        mat.flush()


def allocate(n: int, dtype, on_disk: bool, path: str | None) -> np.ndarray:
    """Create an uninitialised n x n matrix, on disk or in RAM."""
    if on_disk:
        assert path is not None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return np.memmap(path, dtype=dtype, mode="w+", shape=(n, n))
    return np.empty((n, n), dtype=dtype)


def open_existing(n: int, dtype, path: str, mode: str = "r") -> np.memmap:
    return np.memmap(path, dtype=dtype, mode=mode, shape=(n, n))


def generate(
    n: int,
    dtype,
    on_disk: bool,
    path: str | None,
    seed: int,
    fill: str = "random",
    scale: float = 1.0,
    data_rank: int | None = None,
    spectral_alpha: float = 1.0,
) -> np.ndarray:
    if data_rank is not None and (
        isinstance(data_rank, bool) or not isinstance(data_rank, Integral)
        or data_rank < 1
    ):
        raise ValueError("data_rank must be a positive integer or None")
    if fill == "decaying-spectrum" and (
        isinstance(spectral_alpha, bool) or not isinstance(spectral_alpha, Real)
        or not math.isfinite(float(spectral_alpha)) or spectral_alpha < 0
    ):
        # A non-finite alpha makes the k**-alpha weights all-NaN (all-NaN matrix)
        # or, for Inf, collapses them to a silent rank 1 -- corrupting the
        # benchmark input, so reject it at this public boundary.
        raise ValueError(
            f"spectral_alpha must be a finite number >= 0, got {spectral_alpha!r}")
    mat = allocate(n, dtype, on_disk, path)
    if fill == "random":
        _fill_random(mat, seed, scale)
    elif fill == "iota":
        _fill_iota(mat, seed)
    elif fill == "zeros":
        mat[:] = 0
    elif fill == "lowrank":
        r = data_rank if data_rank is not None else max(1, n // 32)
        _fill_lowrank(mat, seed, min(r, n))
    elif fill == "decaying-spectrum":
        # spectral_alpha was already validated at the top of generate() (the
        # stronger check that also rejects non-Real / bool), so no re-check here.
        r = data_rank if data_rank is not None else max(1, n // 32)
        _fill_decaying_spectrum(mat, seed, min(r, n), spectral_alpha)
    else:
        raise ValueError(f"unknown fill {fill!r}")
    return mat


def bytes_human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PiB"


def should_use_disk(n: int, item_bytes: int, storage: str, host_free: int) -> bool:
    """Decide RAM vs disk for 'auto': three n x n matrices must fit comfortably
    (<= 50% of free RAM) to stay in RAM."""
    if storage == "ram":
        return False
    if storage == "disk":
        return True
    need = 3 * n * n * item_bytes
    return need > 0.5 * host_free
