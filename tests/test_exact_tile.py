"""CPU-only tests for multiply_exact tile sizing (Fixes #144).

The per-(row, k) working set is acc (T×n) + Ar (T×T) + Bk (T×n) + matmul
output (T×n) while the in-place add completes: T·(3n + T) elements.

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


def _working_set_elems(t: int, n: int) -> int:
    return t * (3 * n + t)


def test_exact_tile_within_budget():
    n = 8192
    item = 4
    frac = 0.6
    free = 512 * 1024**2
    backend = _FakeBackend(free)
    t = _exact_tile(n, backend, item, frac)
    budget_elems = int(free * frac) // item
    assert _working_set_elems(t, n) <= budget_elems


def test_exact_tile_at_least_one():
    backend = _FakeBackend(1024)
    assert _exact_tile(100000, backend, 4, 0.3) >= 1


def test_exact_tile_never_exceeds_n():
    backend = _FakeBackend(1024**4)
    n = 256
    assert _exact_tile(n, backend, 4, 0.5) <= n


def test_exact_tile_old_picker_would_overshoot_budget():
    # Regression for #144: the legacy T²+2nT model picked T ~1.7x too large once
    # the GEMM output is counted in the real working set T·(3n+T).
    n = 8192
    item = 4
    frac = 0.6
    free = 1 << 30
    budget_elems = int(free * frac) // item
    t_old = int((math.sqrt(4 * n * n + 4 * budget_elems) - 2 * n) / 2)
    t_old = max(1, min(t_old, n))
    assert t_old * (3 * n + t_old) > budget_elems
    t_new = _exact_tile(n, _FakeBackend(free), item, frac)
    assert _working_set_elems(t_new, n) <= budget_elems


def test_exact_tile_issue_repro_table_row():
    n = 4096
    item = 4
    frac = 0.3
    free = 256 * 1024**2
    backend = _FakeBackend(free)
    t = _exact_tile(n, backend, item, frac)
    budget_elems = int(free * frac) // item
    ratio = _working_set_elems(t, n) / budget_elems
    assert ratio <= 1.0 + 1e-9


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
