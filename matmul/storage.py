"""Matrix storage: in-RAM ndarrays or on-disk memory-mapped files.

A memmap lets us address a 65 GB (128k x 128k fp32) matrix without ever
holding it fully in RAM; slicing a tile pulls only that tile off disk.
"""
from __future__ import annotations

import os
import numpy as np


def _fill_random(mat: np.ndarray, seed: int, scale: float = 1.0) -> None:
    """Fill a (possibly memmapped) n x n matrix with N(0, scale^2) values,
    one row-block at a time so we never materialise the whole thing in RAM."""
    n = mat.shape[0]
    rng = np.random.default_rng(seed)
    # Keep each generated block near ~256 MiB of fp64 temporaries.
    block = max(1, min(n, (256 * 1024**2) // (n * 8)))
    for r0 in range(0, n, block):
        r1 = min(n, r0 + block)
        chunk = rng.standard_normal((r1 - r0, n)) * scale
        mat[r0:r1, :] = chunk.astype(mat.dtype, copy=False)
    if isinstance(mat, np.memmap):
        mat.flush()


def _fill_iota(mat: np.ndarray) -> None:
    """Cheap deterministic fill (value = (i+j) mod 97) for fast benchmarking."""
    n = mat.shape[0]
    block = max(1, min(n, (256 * 1024**2) // (n * 8)))
    cols = np.arange(n)
    for r0 in range(0, n, block):
        r1 = min(n, r0 + block)
        rows = np.arange(r0, r1)[:, None]
        mat[r0:r1, :] = ((rows + cols) % 97).astype(mat.dtype, copy=False)
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
        _fill_iota(mat)
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
