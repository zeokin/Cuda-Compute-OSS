"""Matrix storage: in-RAM ndarrays or on-disk memory-mapped files.

A memmap lets us address a 65 GB (128k x 128k fp32) matrix without ever
holding it fully in RAM; slicing a tile pulls only that tile off disk.
"""
from __future__ import annotations

import os
import numpy as np


def _fill_random(mat: np.ndarray, seed: int, scale: float = 1.0) -> None:
    """Fill a (possibly memmapped) n x n matrix with N(0, scale^2) values,
    one row-block at a time so we never materialise the whole thing in RAM.

    ``chunk`` is scaled in place: ``standard_normal`` already returns float64, so
    an out-of-place ``* scale`` keeps the raw draw alive alongside the scaled
    copy and costs TWO full row-blocks, doubling the ~256 MiB this loop budgets
    for one. ``*=`` is bit-identical to ``*``, and this is the DEFAULT fill --
    the out-of-core (128k x 128k) regime this module exists for is exactly where
    host RAM is tightest.
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
) -> np.ndarray:
    mat = allocate(n, dtype, on_disk, path)
    if fill == "random":
        _fill_random(mat, seed, scale)
    elif fill == "iota":
        _fill_iota(mat, seed)
    elif fill == "zeros":
        mat[:] = 0
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
