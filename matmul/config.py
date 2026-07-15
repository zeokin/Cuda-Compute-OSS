"""Configuration for the matrix-multiplication system."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import numpy as np

# Supported element types. NumPy has no native bf16, so we expose the three
# types that both NumPy and PyTorch handle directly.
DTYPES = {
    "fp16": np.float16,
    "fp32": np.float32,
    "fp64": np.float64,
}


@dataclass
class Config:
    """All knobs for a run.

    device        : GPU device index (CUDA).
    dtype         : element type, one of DTYPES keys.
    tile          : tile edge length T. None => auto-picked from free VRAM.
    vram_fraction : fraction of *free* device memory the tiles may occupy.
    accumulate_fp32: for fp16 inputs, accumulate the tile sum in fp32 (accuracy).
    storage       : "ram" | "disk" | "auto". Where A/B/C live (host-side).
    workdir       : directory for memmap files when storage != "ram".
    seed          : RNG seed for generated matrices.
    verbose       : print progress / summary lines.

    Compute always runs on a GPU (CUDA/MPS) via PyTorch; see ``backend.py``.
    """

    device: int = 0
    dtype: str = "fp32"
    tile: int | None = None
    vram_fraction: float = 0.6
    accumulate_fp32: bool = True
    force_tiled: bool = False
    storage: str = "auto"
    workdir: str = "./_matmul_data"
    seed: int = 0
    verbose: bool = True

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(
                f"dtype must be one of {list(DTYPES)}, got {self.dtype!r}"
            )
        if (isinstance(self.device, bool) or not isinstance(self.device, Integral)
                or self.device < 0):
            raise ValueError("device must be a non-negative integer")
        if (isinstance(self.seed, bool) or not isinstance(self.seed, Integral)
                or self.seed < 0):
            raise ValueError("seed must be a non-negative integer")
        if not (0.0 < self.vram_fraction <= 0.95):
            raise ValueError("vram_fraction must be in (0, 0.95]")
        if self.tile is not None:
            # ``range`` requires an integer step. A float can pass a simple
            # positivity check here but then fails later in gemm._tiles, after
            # device setup. Bool is an Integral subclass but is not a useful
            # tile size. Reject both cases at the public configuration boundary.
            if (isinstance(self.tile, bool) or not isinstance(self.tile, Integral)
                    or self.tile < 1):
                # A non-positive tile yields an empty tiling schedule in
                # gemm._tiles (range(0, n, T) is empty for T <= 0), so the tiled
                # loop never runs and C is returned uninitialised. Reject it up
                # front; use None to auto-pick T from free VRAM.
                raise ValueError("tile must be a positive integer or None")
        if self.storage not in ("ram", "disk", "auto"):
            raise ValueError("storage must be ram|disk|auto")

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(DTYPES[self.dtype])

    @property
    def item_bytes(self) -> int:
        return self.np_dtype.itemsize

    @property
    def acc_dtype(self) -> np.dtype:
        """Dtype used to accumulate the sum over K-tiles."""
        if self.np_dtype == np.float16 and self.accumulate_fp32:
            return np.dtype(np.float32)
        return self.np_dtype
