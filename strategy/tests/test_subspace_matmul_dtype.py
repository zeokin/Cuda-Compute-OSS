"""CPU-only tests for strategy.subspace_matmul()'s dtype validation (issue #222).

subspace_matmul() resolves its dtype and checks A/B *before* the GPU backend is
built, so these raise on any machine, GPU or not. Previously _dtype_name silently
returned "fp32" for an unsupported dtype (int, uint, ...), so subspace_matmul on
integer arrays crashed deep inside torch.matmul, and an A/B dtype mismatch was
never caught. This is the strategy-package counterpart of matmul's #209.

Run:  python strategy/tests/test_subspace_matmul_dtype.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import DTYPES, _dtype_name, subspace_matmul


def _square(dtype, n=8):
    return np.ones((n, n), dtype=dtype)


def test_dtype_name_rejects_unsupported():
    for bad in (np.int32, np.int64, np.uint8, np.complex128):
        try:
            _dtype_name(bad)
        except ValueError as e:
            assert "unsupported dtype" in str(e)
        else:
            raise AssertionError(f"_dtype_name({np.dtype(bad)}) should raise ValueError")


def test_dtype_name_accepts_the_three_supported():
    assert _dtype_name(np.float16) == "fp16"
    assert _dtype_name(np.float32) == "fp32"
    assert _dtype_name(np.float64) == "fp64"
    assert set(DTYPES) == {"fp16", "fp32", "fp64"}


def test_subspace_matmul_rejects_integer_dtype_before_backend():
    # No GPU here: this must be the validation ValueError, never a backend
    # RuntimeError or a torch.matmul error.
    try:
        subspace_matmul(_square(np.int64), _square(np.int64))
    except ValueError as e:
        assert "unsupported dtype" in str(e)
    else:
        raise AssertionError("integer inputs should raise ValueError")


def test_subspace_matmul_rejects_dtype_mismatch_before_backend():
    try:
        subspace_matmul(_square(np.float16), _square(np.float32))
    except ValueError as e:
        assert "same dtype" in str(e)
    else:
        raise AssertionError("mismatched dtypes should raise ValueError")


def test_subspace_matmul_shape_guard_still_fires_first():
    # A non-square input must still be rejected by the pre-existing shape guard.
    A = np.ones((8, 5), dtype=np.float32)
    try:
        subspace_matmul(A, A)
    except ValueError as e:
        assert "square" in str(e)
    else:
        raise AssertionError("non-square input should raise ValueError")


if __name__ == "__main__":
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
