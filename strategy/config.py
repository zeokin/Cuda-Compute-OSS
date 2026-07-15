"""Configuration for the subspace ('smart') matrix-multiplication strategy.

Standalone: this package does not import from the sibling `matmul` package.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import numpy as np

DTYPES = {
    "fp16": np.float16,
    "fp32": np.float32,
    "fp64": np.float64,
}


@dataclass
class Config:
    """Knobs for the subspace strategy.

    device        : GPU device index (CUDA).
    dtype         : element type, one of DTYPES keys.
    rank_m        : subspace dimension M. None => min(n, max(64, n // 8)). Smaller = faster,
                    less accurate. The approximation is exact only when M = n
                    or when A/B live in the captured subspace (low rank).
    transform     : registry name ("rsvd", the only built-in) OR a Transform
                    instance (the pluggable "core tech").
    transform_seed: RNG seed for randomized transforms.
    vram_fraction : fraction of free device memory the streamed blocks may use.
    storage       : "ram" | "disk" | "auto" for the generated matrices.
    workdir       : directory for memmap files when storage != "ram".
    seed          : RNG seed for generated matrices.
    verbose       : print progress / summary lines.

    Compute always runs on a GPU (CUDA/MPS) via PyTorch; see ``backend.py``.
    """

    device: int = 0
    dtype: str = "fp32"
    rank_m: int | None = None
    transform: object = "rsvd"
    transform_seed: int = 0
    vram_fraction: float = 0.6
    storage: str = "auto"
    workdir: str = "./_strategy_data"
    seed: int = 0
    verbose: bool = True

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}, got {self.dtype!r}")
        if (isinstance(self.device, bool) or not isinstance(self.device, Integral)
                or self.device < 0):
            raise ValueError("device must be a non-negative integer")
        for name in ("transform_seed", "seed"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if not (0.0 < self.vram_fraction <= 0.95):
            raise ValueError("vram_fraction must be in (0, 0.95]")
        if self.rank_m is not None and (
            isinstance(self.rank_m, bool) or not isinstance(self.rank_m, Integral)
        ):
            # A fractional rank passes the range guard in multiply_subspace()
            # but then fails deep in a shape/range operation. ``bool`` is an
            # Integral subclass, yet rank True/False is never meaningful.
            raise ValueError("rank_m must be an integer or None")
        if self.storage not in ("ram", "disk", "auto"):
            raise ValueError("storage must be ram|disk|auto")

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(DTYPES[self.dtype])

    @property
    def item_bytes(self) -> int:
        return self.np_dtype.itemsize

    @property
    def compute_dtype(self) -> np.dtype:
        """Projection math is done in fp32 for fp16 inputs (accuracy)."""
        if self.np_dtype == np.float16:
            return np.dtype(np.float32)
        return self.np_dtype
