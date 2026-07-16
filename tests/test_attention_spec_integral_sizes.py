"""AttentionSpec must accept NumPy integer sizes (issue #254).

The size fields are type-checked before their range check. That check used the
bare `int`, but a size derived from NumPy -- an array's .shape entry, an
np.arange element, `n // 8` on an np.int64 -- is an np.integer: NOT an `int`,
though it IS a numbers.Integral. So AttentionSpec(seq=np.int64(4096)) raised
"seq must be an int, got int64" for exactly the integers callers compute.

Testing against numbers.Integral accepts np.int32/np.int64 while still rejecting
float, bool and str. Pure dataclass validation; no torch/GPU needed.

Run:  python tests/test_attention_spec_integral_sizes.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention.spec import AttentionSpec

_SIZE_FIELDS = ("batch", "heads", "seq", "dim", "window", "seed")


def test_numpy_integer_sizes_are_accepted():
    for dtype in (np.int32, np.int64):
        spec = AttentionSpec(batch=dtype(1), heads=dtype(2), seq=dtype(16),
                             dim=dtype(8), window=dtype(4), seed=dtype(0))
        assert int(spec.seq) == 16 and int(spec.window) == 4


def test_each_size_field_accepts_a_numpy_integer():
    defaults = dict(batch=1, heads=2, seq=16, dim=8, window=4, seed=0)
    for name in _SIZE_FIELDS:
        kwargs = dict(defaults)
        kwargs[name] = np.int64(defaults[name])
        AttentionSpec(**kwargs)  # must not raise


def test_a_size_from_a_numpy_shape_is_accepted():
    """The realistic path: a size read straight off a NumPy array."""
    v = np.zeros((1, 2, 16, 8), dtype=np.float32)
    idx = np.arange(16)
    AttentionSpec(batch=np.int64(v.shape[0]), heads=np.int64(v.shape[1]),
                  seq=np.int64(v.shape[2]), dim=np.int64(v.shape[3]),
                  window=idx[4], seed=np.int64(0))


def test_floats_bools_and_strings_are_still_rejected():
    for bad in (2.5, np.float32(4.0), True, "16"):
        try:
            AttentionSpec(seq=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AttentionSpec(seq={bad!r}) should raise ValueError")


def test_range_checks_still_apply_to_numpy_integers():
    for bad in (np.int64(0), np.int64(-1)):
        try:
            AttentionSpec(seq=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AttentionSpec(seq={bad!r}) should raise ValueError")


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
