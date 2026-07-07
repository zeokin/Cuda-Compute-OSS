"""Shared benchmark spec for the attention playground."""
from __future__ import annotations

from dataclasses import asdict, dataclass


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

    def as_dict(self) -> dict:
        return asdict(self)
