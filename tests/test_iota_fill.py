"""CPU-only tests for iota fill seed threading (issue #104).

Run:  python tests/test_iota_fill.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul import storage as matmul_storage
from strategy import storage as strategy_storage


def test_iota_fill_differs_by_seed_strategy():
    n = 64
    a0 = strategy_storage.generate(n, np.float32, False, None, seed=0, fill="iota")
    a1 = strategy_storage.generate(n, np.float32, False, None, seed=1, fill="iota")
    assert not np.array_equal(a0, a1)


def test_iota_fill_differs_by_seed_matmul():
    n = 64
    a0 = matmul_storage.generate(n, np.float32, False, None, seed=0, fill="iota")
    a1 = matmul_storage.generate(n, np.float32, False, None, seed=1, fill="iota")
    assert not np.array_equal(a0, a1)


def test_iota_eval_couple_operands_differ():
    """Mirror eval/evaluator.py seed offsets: A and B must not be identical."""
    n = 128
    seed = 7
    for i in range(3):
        a = strategy_storage.generate(
            n, np.float32, False, None, seed + 2 * i, fill="iota"
        )
        b = strategy_storage.generate(
            n, np.float32, False, None, seed + 2 * i + 1, fill="iota"
        )
        assert not np.array_equal(a, b), f"couple {i}: A == B with iota fill"


def test_iota_eval_couples_differ():
    """Distinct couples for --pairs averaging (eval harness contract)."""
    n = 128
    seed = 7
    couples = []
    for i in range(3):
        a = strategy_storage.generate(
            n, np.float32, False, None, seed + 2 * i, fill="iota"
        )
        couples.append(a.copy())
    assert not np.array_equal(couples[0], couples[1])
    assert not np.array_equal(couples[1], couples[2])


def test_iota_same_seed_is_deterministic():
    n = 32
    a = strategy_storage.generate(n, np.float32, False, None, seed=42, fill="iota")
    b = strategy_storage.generate(n, np.float32, False, None, seed=42, fill="iota")
    assert np.array_equal(a, b)


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
