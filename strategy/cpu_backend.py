"""NumPy-backed stand-in for strategy.backend.Backend -- SMOKE-TEST ONLY.

Satisfies the same surface (xp / to_device / to_host / matmul / synchronize /
free_compute_bytes / host_available_bytes / name) so a Transform's basis()
can be exercised with zero GPU/torch dependency, for docs/testing-strategy.md's
Tier-1 local check (strategy/smoke.py). Never used to produce a scored
result -- eval/evaluator.py always requires the real GPU backend and will
refuse to run without one.
"""
from __future__ import annotations

import numpy as np

from .backend import _host_available_bytes


class _NumpyXP:
    """Minimal namespace matching strategy.backend._TorchXP's surface,
    backed directly by NumPy -- no device concept, no torch dependency."""

    def __init__(self):
        self.linalg = np.linalg
        self.float16, self.float32, self.float64 = np.float16, np.float32, np.float64

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def empty(self, shape, dtype=None):
        return np.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None):
        return np.full(shape, fill_value, dtype=dtype)

    def arange(self, *args, dtype=None, **kw):
        return np.arange(*args, dtype=dtype, **kw)

    def cos(self, x):
        return np.cos(x)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(list(arrays), axis=axis)


class CPUBackend:
    """NumPy-backed stand-in for Backend. Same surface, CPU only, smoke-test
    only -- see module docstring."""

    def __init__(self, device: int = 0, verbose: bool = True):
        self.device_id = int(device)
        self.verbose = verbose
        self.kind = "cpu"
        self.gpu = False
        self.xp = _NumpyXP()

    @property
    def name(self) -> str:
        return "CPU (NumPy, smoke-test only -- not a scoring backend)"

    def free_compute_bytes(self) -> int:
        return self.host_available_bytes()

    def host_available_bytes(self) -> int:
        return _host_available_bytes()

    def to_device(self, host_array):
        return np.ascontiguousarray(host_array)

    def to_host(self, dev_array) -> np.ndarray:
        return np.asarray(dev_array)

    def zeros(self, shape, dtype):
        return self.xp.zeros(shape, dtype=dtype)

    def matmul(self, a, b):
        return a @ b

    def synchronize(self):
        pass
