"""CPU-only tests for storage.should_use_disk working-set budgeting.

run() holds 3 n x n matrices (A, B, C); compare() holds 4 (A, B, Ce, Cs). The
auto RAM-vs-disk decision must budget for the real count, or compare() picks RAM
for a footprint that only fits on disk.

Run:  python strategy/tests/test_storage_budget.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import storage


def test_default_working_set_is_three_matrices():
    n, item = 100, 8
    per_matrix = n * n * item
    # host_free chosen so exactly 3 matrices sit at the 50% line: 3*per = 0.5*free.
    host_free = int(2 * 3 * per_matrix)
    # 3 matrices == 50% -> not strictly greater -> stay in RAM.
    assert storage.should_use_disk(n, item, "auto", host_free) is False
    # 4 matrices > 50% -> go to disk.
    assert storage.should_use_disk(n, item, "auto", host_free, n_matrices=4) is True


def test_compare_budget_flips_ram_to_disk_in_the_gap():
    """The window where 3 fit but 4 don't: run() stays RAM, compare() must not."""
    n, item = 100, 8
    per_matrix = n * n * item
    # free RAM sits between 6*per (3 matrices at 50%) and 8*per (4 at 50%).
    host_free = int(7 * per_matrix)
    run_decision = storage.should_use_disk(n, item, "auto", host_free)          # 3 matrices
    compare_decision = storage.should_use_disk(n, item, "auto", host_free, n_matrices=4)
    assert run_decision is False, "3-matrix run() should still fit in RAM here"
    assert compare_decision is True, "4-matrix compare() must spill to disk here"


def test_explicit_storage_modes_ignore_matrix_count():
    n, item = 100, 8
    assert storage.should_use_disk(n, item, "ram", 1, n_matrices=4) is False
    assert storage.should_use_disk(n, item, "disk", 10**18, n_matrices=4) is True


def test_compare_passes_four_matrices(monkeypatch):
    """compare() must call should_use_disk with n_matrices=4 (integration guard)."""
    import numpy as np

    from strategy import runner
    from strategy.config import Config

    seen = {}

    def spy(n_, item_, storage_, host_free_, n_matrices=3):
        seen["n_matrices"] = n_matrices
        return False  # force RAM so no files are created

    class _FakeBackend:
        name = "fake"

        def synchronize(self):
            pass

        def host_available_bytes(self):
            return 10**12

    monkeypatch.setattr(runner.storage, "should_use_disk", spy)
    monkeypatch.setattr(runner, "Backend", lambda *a, **k: _FakeBackend())
    monkeypatch.setattr(runner.storage, "generate",
                        lambda n_, dt_, on_disk, path, *a, **k: np.zeros((n_, n_), dtype=dt_))
    monkeypatch.setattr(runner.storage, "allocate",
                        lambda n_, dt_, on_disk, path: np.zeros((n_, n_), dtype=dt_))
    monkeypatch.setattr(runner.subspace, "multiply_exact",
                        lambda *a, **k: {"mode": "exact", "flop_exact": 1})
    monkeypatch.setattr(runner.subspace, "multiply_subspace",
                        lambda *a, **k: {"mode": "smart", "flop_exact": 1, "flop_actual": 1})

    runner.compare(4, Config(workdir=".", storage="auto", verbose=False))
    assert seen["n_matrices"] == 4


if __name__ == "__main__":
    try:
        import pytest
    except ImportError:
        # The pure should_use_disk tests need no pytest; run them directly.
        fns = [v for k, v in sorted(globals().items())
               if k.startswith("test_") and k != "test_compare_passes_four_matrices"]
        failed = 0
        for fn in fns:
            try:
                fn()
                print(f"PASS  {fn.__name__}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {fn.__name__}: {e}")
        print(f"\n{len(fns) - failed}/{len(fns)} passed (compare-integration test needs pytest)")
        sys.exit(1 if failed else 0)

    raise SystemExit(pytest.main([__file__, "-v"]))
