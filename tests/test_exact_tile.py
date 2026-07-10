"""CPU-only tests for multiply_exact tile sizing (issue #144).

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


def _budget_elems(free: int, frac: float, item: int) -> int:
    return max(1, int(free * frac) // item)


def test_exact_tile_within_budget():
    """Picked T must fit acc + Ar + Bk + matmul output: T*(3n + T)."""
    n = 8192
    item = 4
    frac = 0.6
    free = 1 << 30
    backend = _FakeBackend(free)
    t = _exact_tile(n, backend, item, frac)
    budget_elems = _budget_elems(free, frac, item)
    assert t * (3 * n + t) <= budget_elems


def test_exact_tile_at_least_one():
    backend = _FakeBackend(1024)
    assert _exact_tile(100000, backend, 4, 0.3) >= 1


def test_exact_tile_never_exceeds_n():
    backend = _FakeBackend(1024**4)
    n = 256
    assert _exact_tile(n, backend, 4, 0.5) <= n


def test_old_model_would_overshoot():
    """Witness: the pre-#144 model (2n + T per row) overshot at n=8192."""
    n, item, frac = 8192, 4, 0.6
    free = 1 << 30
    budget_elems = _budget_elems(free, frac, item)
    old_t = int((math.sqrt(4 * n * n + 4 * budget_elems) - 2 * n) / 2)
    assert old_t * (2 * n + old_t) <= budget_elems
    assert old_t * (3 * n + old_t) > budget_elems


def test_issue_144_table_case_256mb():
    n, item, frac = 4096, 4, 0.3
    free = 256 * 1024**2
    t = _exact_tile(n, _FakeBackend(free), item, frac)
    budget_elems = _budget_elems(free, frac, item)
    assert t * (3 * n + t) <= budget_elems


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
