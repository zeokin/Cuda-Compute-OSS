"""Shared benchmark spec for the attention playground."""
from __future__ import annotations

import math
import numbers
import re
from dataclasses import asdict, dataclass
from numbers import Real

# Kept in sync with attention.data.torch_dtype's map and the benchmark CLI's
# --dtype choices.
DTYPES = ("fp16", "fp32", "fp64")
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")


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
        # Check type before range: an int-typed field holding e.g. a float
        # (window=2.5) can pass every range check below yet still crash later,
        # deep inside tensor slicing, only for some (seq, window, block_size)
        # combinations -- confusing and data-shape-dependent. bool is excluded
        # even though isinstance(True, int) is True: a bool value here is
        # never meaningful.
        #
        # Test against numbers.Integral, not the bare `int`: a size derived from
        # NumPy (an array's .shape entry, an np.arange element, n // 8 on an
        # np.int64) is an np.integer, which is NOT an `int` but IS Integral. A
        # bare isinstance(value, int) rejects those with "seq must be an int,
        # got int64" even though they are exactly the integers callers compute.
        for name in ("batch", "heads", "seq", "dim", "window", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{name} must be an int, got {type(value).__name__}")
        for name in ("batch", "heads", "seq", "dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.window < 0:
            raise ValueError("window must be >= 0")
        if not isinstance(self.causal, bool):
            raise ValueError("causal must be a bool")
        for name in ("local_weight", "global_weight", "freq_decay"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite real number")
        if self.local_weight < 0 or self.global_weight < 0:
            raise ValueError("branch weights must be >= 0")
        if self.local_weight + self.global_weight <= 0:
            raise ValueError("at least one branch weight must be positive")
        if self.freq_decay < 0:
            raise ValueError("freq_decay must be >= 0")
        # dtype/device used to fail late and opaquely -- a bad dtype as a
        # KeyError inside data.torch_dtype, a bad device as a raw PyTorch
        # RuntimeError inside resolve_device -- both well after construction
        # and, for device, after generate_qkv had already started building
        # tensors. Reject both here instead.
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if not _DEVICE_RE.match(self.device):
            raise ValueError(
                "device must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:<index>', "
                f"got {self.device!r}"
            )

    def as_dict(self) -> dict:
        return asdict(self)
