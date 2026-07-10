"""Shared benchmark spec for the attention playground."""
from __future__ import annotations

from dataclasses import asdict, dataclass

_DTYPES = ("fp16", "fp32", "fp64")


@dataclass(frozen=True)
class AttentionSpec:
    batch: int = 1
    heads: int = 8
    seq: int = 4096
    dim: int = 64
    dtype: str = "fp16"
    window: int = 256
    local_weight: float = 0.85
    global_weight: float = 0.15
    freq_decay: float = 1.0
    causal: bool = False
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in ("batch", "heads", "seq", "dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        # Validate dtype at the spec boundary (like every other field) instead of
        # letting an unknown value fall through to an opaque KeyError deep in
        # data.torch_dtype() when generate_qkv() finally runs.
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
        if self.window < 0:
            raise ValueError("window must be >= 0")
        if self.local_weight < 0 or self.global_weight < 0:
            raise ValueError("branch weights must be >= 0")
        if self.local_weight + self.global_weight <= 0:
            raise ValueError("at least one branch weight must be positive")
        if self.freq_decay < 0:
            raise ValueError("freq_decay must be >= 0")

    def as_dict(self) -> dict:
        return asdict(self)
