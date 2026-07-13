"""CPU-only tests for multiply_exact tile sizing.

Run:  python tests/test_exact_tile.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.subspace import _exact_tile


class _FakeBackend:
    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free


def test_exact_tile_within_budget():
    n = 8192
    item = 4
    frac = 0.6
    free = 512 * 1024**2
    backend = _FakeBackend(free)
    t = _exact_tile(n, backend, item, frac)
    budget_elems = int(free * frac) // item
    # _exact_tile sizes against the real per-(row, k) working set T*(3n+T)
    # (acc + A panel + B panel + the GEMM output), i.e. it solves T^2 + 3nT =
    # budget (see #144). Assert that exact bound: the weaker T*(2n+T) <= budget
    # would still pass even if the code regressed to the old, OOM-prone model.
    assert t * (3 * n + t) <= budget_elems


def test_exact_tile_at_least_one():
    backend = _FakeBackend(1024)
    assert _exact_tile(100000, backend, 4, 0.3) >= 1


def test_exact_tile_never_exceeds_n():
    backend = _FakeBackend(1024**4)
    n = 256
    assert _exact_tile(n, backend, 4, 0.5) <= n


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
