"""CPU-only tests for matrix generation (no GPU needed).

These pin the seed contract of ``storage.generate``: distinct seeds must yield
distinct matrices for EVERY fill, so the eval harness's per-couple seeds
(A=seed+2i, B=seed+2i+1) actually produce independent A/B and distinct couples.
The ``iota`` fill regressed this by dropping the seed, collapsing every couple to
an identical, symmetric A@A.

Run:  python strategy/tests/test_storage.py   (or via pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import storage


def _gen(seed, fill, n=64, dtype=np.float32):
    return storage.generate(n, dtype, False, None, seed, fill)


def test_iota_seed_makes_distinct_matrices():
    """The regression: A=generate(seed) and B=generate(seed+1) must differ."""
    A = _gen(0, "iota")
    B = _gen(1, "iota")
    assert not np.array_equal(A, B), "iota ignores the seed: A == B (couple collapse)"


def test_iota_is_deterministic_per_seed():
    assert np.array_equal(_gen(7, "iota"), _gen(7, "iota"))


def test_iota_distinct_across_pairs():
    """Every (A_i, B_i) couple in an eval sweep must be distinct."""
    mats = [_gen(2 * i, "iota") for i in range(4)] + [_gen(2 * i + 1, "iota") for i in range(4)]
    for a in range(len(mats)):
        for b in range(a + 1, len(mats)):
            assert not np.array_equal(mats[a], mats[b]), f"iota couples {a},{b} identical"


def test_iota_pattern_value():
    """Values stay the cheap deterministic (i+j+seed) mod 97 pattern."""
    n, seed = 8, 3
    A = _gen(seed, "iota", n=n)
    rows = np.arange(n)[:, None]
    cols = np.arange(n)
    expected = ((rows + cols + seed) % 97).astype(np.float32)
    assert np.array_equal(A, expected)


def test_random_and_lowrank_still_seed_dependent():
    assert not np.array_equal(_gen(0, "random"), _gen(1, "random"))
    A = storage.generate(64, np.float32, False, None, 0, "lowrank", data_rank=4)
    B = storage.generate(64, np.float32, False, None, 1, "lowrank", data_rank=4)
    assert not np.array_equal(A, B)


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
