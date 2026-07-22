"""Tests for Backend.to_host's pinned bounce buffer.

Every result the strategy produces leaves the device through ``to_host`` --
reconstruct's (rb, n) output rows and multiply_exact's accumulator -- so that
copy, not the math, bounds the engine. Pageable device->host runs at roughly a
seventh of pinned bandwidth, so ``to_host`` streams large reads through one
small, reused pinned buffer.

The bounce itself needs a real device (pin_memory requires CUDA), so those
checks skip cleanly without one -- exactly like the rest of strategy/tests. The
threshold/constant checks are pure CPU and always run.

Run:  python strategy/tests/test_pinned_host_staging.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import backend as backend_mod


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()
                    or (getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()))
    except Exception:  # noqa: BLE001
        return False


HAVE_GPU = _gpu_available()

try:
    import pytest
    pytestmark = pytest.mark.skipif(
        not HAVE_GPU, reason="no CUDA/MPS GPU; pin_memory needs a real device")
except ImportError:
    pass


def test_pinned_thresholds_are_sane():
    # Pure CPU: the bounce must only kick in for reads big enough to pay for it,
    # and the chunk must never exceed what we're willing to pin up front.
    assert backend_mod._PINNED_MIN_BYTES > 0
    assert backend_mod._PINNED_CHUNK_BYTES > 0
    assert backend_mod._PINNED_MIN_BYTES <= backend_mod._PINNED_CHUNK_BYTES


def test_to_host_is_bit_identical_to_the_pageable_copy():
    """The bounce must not change a single bit, at any dtype/shape, above and
    below the threshold, and for non-contiguous inputs."""
    if not HAVE_GPU:
        return
    import torch

    bk = backend_mod.Backend(0, False)
    for dtype in (torch.float32, torch.float64, torch.float16):
        for shape in [(2048, 2048), (1024, 3000), (17, 3), (1, 1)]:
            t = torch.randn(*shape, device=bk.dev, dtype=dtype)
            got = bk.to_host(t)
            want = t.detach().to("cpu").numpy()
            assert got.dtype == want.dtype, (dtype, shape)
            assert got.shape == want.shape, (dtype, shape)
            assert np.array_equal(got, want), (dtype, shape)

    # A transposed view is non-contiguous; to_host must still match.
    t = torch.randn(2048, 2048, device=bk.dev).T
    assert np.array_equal(bk.to_host(t), t.detach().to("cpu").numpy())


def test_large_read_uses_and_reuses_one_pinned_buffer():
    """The buffer is allocated once and reused -- pinning per transfer would
    hand back most of the speedup it exists to gain."""
    if not HAVE_GPU:
        return
    import torch

    bk = backend_mod.Backend(0, False)
    assert bk._pin_buf is None                       # nothing pinned until needed

    big = torch.randn(4096, 4096, device=bk.dev)     # 64 MiB >> _PINNED_MIN_BYTES
    bk.to_host(big)
    first = bk._pin_buf
    assert first is not None and first.is_pinned()
    # capped by the chunk size, not the size of the read
    assert first.numel() * first.element_size() <= backend_mod._PINNED_CHUNK_BYTES

    bk.to_host(big)
    assert bk._pin_buf is first, "pinned buffer must be reused across calls"


def test_small_read_skips_the_bounce():
    """Below the threshold to_host takes the direct path and pins nothing."""
    if not HAVE_GPU:
        return
    import torch

    bk = backend_mod.Backend(0, False)
    small = torch.randn(8, 8, device=bk.dev)
    assert np.array_equal(bk.to_host(small), small.detach().to("cpu").numpy())
    assert bk._pin_buf is None


if __name__ == "__main__":
    if not HAVE_GPU:
        print("SKIP  all GPU tests (no CUDA/MPS device)")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
