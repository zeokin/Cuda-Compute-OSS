"""CPU-only tests for the public matmul() dtype validation (issue #208).

matmul()'s dtype/shape checks run *before* the GPU backend is created, so they
raise on any machine, GPU or not. Previously _dtype_name silently mislabelled an
unsupported dtype (int, uint, ...) as "fp32", so matmul(int_A, int_B) crashed deep
inside torch.bmm, and an A/B dtype mismatch was never checked.

Run:  python tests/test_matmul_dtype_validation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul import _dtype_name, matmul, DTYPES


def test_dtype_name_rejects_unsupported():
    for bad in (np.int32, np.int64, np.uint8, np.complex64):
        try:
            _dtype_name(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"_dtype_name({np.dtype(bad)}) should raise ValueError")


def test_dtype_name_accepts_supported():
    assert _dtype_name(np.float16) == "fp16"
    assert _dtype_name(np.float32) == "fp32"
    assert _dtype_name(np.float64) == "fp64"
    # and it agrees with the advertised set
    assert set(DTYPES) == {"fp16", "fp32", "fp64"}


def test_matmul_rejects_integer_dtype_before_backend():
    # No GPU here: this must raise ValueError (validation), never a backend
    # RuntimeError or a torch.bmm error.
    A = np.ones((8, 8), dtype=np.int32)
    B = np.ones((8, 8), dtype=np.int32)
    try:
        matmul(A, B)
    except ValueError as e:
        assert "unsupported dtype" in str(e)
    else:
        raise AssertionError("matmul on int arrays should raise ValueError")


def test_matmul_rejects_dtype_mismatch_before_backend():
    A = np.ones((8, 8), dtype=np.float16)
    B = np.ones((8, 8), dtype=np.float32)
    try:
        matmul(A, B)
    except ValueError as e:
        assert "same dtype" in str(e) or "share a dtype" in str(e)
    else:
        raise AssertionError("matmul on mismatched dtypes should raise ValueError")


def test_matmul_shape_check_still_first():
    # The pre-existing square/shape guard must still fire (and not be shadowed
    # by the new dtype checks) for a non-square input.
    A = np.ones((8, 4), dtype=np.float32)
    B = np.ones((8, 4), dtype=np.float32)
    try:
        matmul(A, B)
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
