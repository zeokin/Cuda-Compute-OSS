"""Backend.free_compute_bytes() must report FREE device memory on MPS.

``torch.mps.recommended_max_memory()`` is Metal's ``recommendedMaxWorkingSetSize``:
a per-device constant, i.e. the TOTAL budget. Returning it verbatim made
free_compute_bytes() answer the same number however much was already resident,
so every budget built on it (``subspace._row_block``, ``subspace._exact_tile``,
``matmul.gemm.auto_tile`` / ``_fits_in_core``) re-spent the whole
``vram_fraction`` on each call and two concurrently live sized allocations
overshot the device.

Both packages carry their own copy of the backend, so this is a parity table:
the rule cannot hold in one and silently lapse in the other.

No GPU and no torch build with MPS is needed -- the tests drive
``Backend.free_compute_bytes`` against a stub ``torch.mps`` namespace, the same
mocking approach tests/test_host_memory.py uses for the host-RAM probes.

Run:  python tests/test_mps_free_memory.py
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul.backend import Backend as MatmulBackend
from strategy.backend import Backend as StrategyBackend

BACKENDS = [("matmul", MatmulBackend), ("strategy", StrategyBackend)]

CEILING = 21 * 1024**3          # a plausible recommendedMaxWorkingSetSize
LIVE = 8 * 1024**3              # bytes held by live tensors


class _FakeDevice:
    type = "mps"


class _FakeMPS:
    """Stub of the ``torch.mps`` namespace with the two memory queries."""

    def __init__(self, ceiling=CEILING, live=0, **absent):
        self._ceiling = ceiling
        self._live = live
        for name in absent:
            setattr(self, name, None)

    def recommended_max_memory(self):
        return self._ceiling

    def current_allocated_memory(self):
        return self._live


class _FakeTorch:
    def __init__(self, mps):
        self.mps = mps


def _backend(cls, mps, host_bytes=4 * 1024**3):
    """A Backend on a fake MPS device -- __init__ would demand a real GPU."""
    backend = cls.__new__(cls)
    backend.device_id = 0
    backend.verbose = False
    backend.torch = _FakeTorch(mps)
    backend.dev = _FakeDevice()
    backend.host_available_bytes = lambda: host_bytes
    return backend


@pytest.mark.parametrize("label,cls", BACKENDS)
def test_live_allocations_are_subtracted(label, cls):
    backend = _backend(cls, _FakeMPS(live=LIVE))
    assert backend.free_compute_bytes() == CEILING - LIVE, label


@pytest.mark.parametrize("label,cls", BACKENDS)
def test_free_memory_shrinks_as_the_device_fills(label, cls):
    # The bug: the reading never moved, so a second sized allocation believed it
    # had the whole budget again. Free bytes must be strictly decreasing.
    mps = _FakeMPS(live=0)
    backend = _backend(cls, mps)
    readings = []
    for live in (0, 4 * 1024**3, 12 * 1024**3):
        mps._live = live
        readings.append(backend.free_compute_bytes())
    assert readings == sorted(readings, reverse=True), label
    assert len(set(readings)) == len(readings), f"{label}: reading never moved"


@pytest.mark.parametrize("label,cls", BACKENDS)
def test_never_reports_negative_free_memory(label, cls):
    # recommendedMaxWorkingSetSize is a soft recommendation; live tensors can
    # exceed it. A negative budget would flow into the tile/row-block pickers.
    backend = _backend(cls, _FakeMPS(live=CEILING * 2))
    assert backend.free_compute_bytes() == 0, label


@pytest.mark.parametrize("label,cls", BACKENDS)
def test_falls_back_to_the_ceiling_when_the_usage_query_is_absent(label, cls):
    # Older torch builds expose recommended_max_memory but not the usage query.
    # Behaviour must degrade to the previous reading, never raise.
    backend = _backend(cls, _FakeMPS(live=LIVE, current_allocated_memory=None))
    assert backend.free_compute_bytes() == CEILING, label


@pytest.mark.parametrize("label,cls", BACKENDS)
def test_falls_back_to_host_ram_without_the_mps_queries(label, cls):
    backend = _backend(cls, _FakeMPS(recommended_max_memory=None), host_bytes=7 * 1024**3)
    assert backend.free_compute_bytes() == 7 * 1024**3, label


@pytest.mark.parametrize("label,cls", BACKENDS)
def test_a_raising_query_falls_back_to_host_ram(label, cls):
    class _Boom(_FakeMPS):
        def current_allocated_memory(self):
            raise RuntimeError("MPS query failed")

    backend = _backend(cls, _Boom(), host_bytes=5 * 1024**3)
    assert backend.free_compute_bytes() == 5 * 1024**3, label


@pytest.mark.parametrize("label,cls", BACKENDS)
def test_cuda_path_still_uses_mem_get_info(label, cls):
    # The CUDA branch already reports true free memory; it must be untouched.
    class _CudaDevice:
        type = "cuda"

    class _FakeCuda:
        @staticmethod
        def mem_get_info(index):
            assert index == 0
            return (3 * 1024**3, 24 * 1024**3)

    backend = _backend(cls, _FakeMPS(live=LIVE))
    backend.dev = _CudaDevice()
    backend.torch.cuda = _FakeCuda()
    assert backend.free_compute_bytes() == 3 * 1024**3, label


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
