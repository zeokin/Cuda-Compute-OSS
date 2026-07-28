"""MPS tile budgets must account for tensors already resident on the GPU."""
from __future__ import annotations

import pytest

from matmul.backend import Backend as MatmulBackend
from strategy.backend import Backend as StrategyBackend


class _Device:
    type = "mps"


class _MPS:
    def __init__(self, ceiling=1000, used=250):
        self.ceiling = ceiling
        self.used = used

    def recommended_max_memory(self):
        return self.ceiling

    def current_allocated_memory(self):
        return self.used


class _Torch:
    def __init__(self, mps):
        self.mps = mps


def _backend(cls, mps):
    backend = cls.__new__(cls)
    backend.device_id = 0
    backend.dev = _Device()
    backend.torch = _Torch(mps)
    backend.host_available_bytes = lambda: 400
    return backend


@pytest.mark.parametrize("Backend", [MatmulBackend, StrategyBackend])
def test_mps_reports_ceiling_minus_live_tensor_memory(Backend):
    mps = _MPS()
    backend = _backend(Backend, mps)
    assert backend.free_compute_bytes() == 750

    mps.used = 600
    assert backend.free_compute_bytes() == 400


@pytest.mark.parametrize("Backend", [MatmulBackend, StrategyBackend])
def test_mps_free_memory_is_never_negative(Backend):
    assert _backend(Backend, _MPS(ceiling=100, used=200)).free_compute_bytes() == 0


@pytest.mark.parametrize("Backend", [MatmulBackend, StrategyBackend])
def test_mps_missing_usage_query_preserves_ceiling_fallback(Backend):
    mps = _MPS()
    mps.current_allocated_memory = None
    assert _backend(Backend, mps).free_compute_bytes() == 1000


@pytest.mark.parametrize("Backend", [MatmulBackend, StrategyBackend])
def test_mps_failing_usage_query_preserves_ceiling_fallback(Backend):
    class FailingUsage(_MPS):
        def current_allocated_memory(self):
            raise RuntimeError("unavailable")

    assert _backend(Backend, FailingUsage()).free_compute_bytes() == 1000
