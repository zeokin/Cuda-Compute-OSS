"""CPU-only coverage for compare()'s four-matrix auto RAM-vs-disk budget (#287).

run() holds A+B+C (3). compare() holds A+B+Ce+Cs (4). The shared
storage.should_use_disk helper still charges 3, so compare must decide disk
itself or it can stay in RAM in the 3-fit/4-don't host-RAM gap and OOM.

Run:  python strategy/tests/test_compare_host_budget.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import storage
from strategy.runner import (
    _COMPARE_RESIDENT_MATRICES,
    _compare_should_use_disk,
)


def test_compare_charges_four_matrices_not_three():
    assert _COMPARE_RESIDENT_MATRICES == 4


def test_gap_where_run_stays_ram_but_compare_spills():
    """host_free between 6*per and 8*per: three matrices fit the 50% line,
    four do not — compare must choose disk while run's helper stays RAM."""
    n, item = 64, 4
    per = n * n * item
    host_free = 7 * per
    assert storage.should_use_disk(n, item, "auto", host_free) is False
    assert _compare_should_use_disk(n, item, "auto", host_free) is True


def test_both_stay_ram_when_four_fit_comfortably():
    n, item = 64, 4
    per = n * n * item
    host_free = 20 * per  # 4*per << 0.5*host_free
    assert storage.should_use_disk(n, item, "auto", host_free) is False
    assert _compare_should_use_disk(n, item, "auto", host_free) is False


def test_both_spill_when_three_already_overflow():
    n, item = 64, 4
    per = n * n * item
    host_free = 4 * per  # even 3*per > 0.5*host_free
    assert storage.should_use_disk(n, item, "auto", host_free) is True
    assert _compare_should_use_disk(n, item, "auto", host_free) is True


def test_explicit_modes_short_circuit():
    assert _compare_should_use_disk(8, 8, "ram", 1) is False
    assert _compare_should_use_disk(8, 8, "disk", 10**18) is True


def test_compare_wires_four_matrix_budget(monkeypatch):
    """Integration: compare() must consult the four-matrix helper, not the
    three-matrix storage.should_use_disk default."""
    import numpy as np

    from strategy import runner
    from strategy.config import Config

    calls = []

    def capture(n_, item_, mode, host_free):
        calls.append((n_, item_, mode, host_free))
        return False

    class _Backend:
        name = "cpu-fake"

        def synchronize(self):
            pass

        def host_available_bytes(self):
            return 10**12

    monkeypatch.setattr(runner, "_compare_should_use_disk", capture)
    monkeypatch.setattr(runner, "Backend", lambda *a, **k: _Backend())
    monkeypatch.setattr(
        runner.storage, "generate",
        lambda n_, dt_, on_disk, path, *a, **k: np.zeros((n_, n_), dtype=dt_),
    )
    monkeypatch.setattr(
        runner.storage, "allocate",
        lambda n_, dt_, on_disk, path: np.zeros((n_, n_), dtype=dt_),
    )
    monkeypatch.setattr(
        runner.subspace, "multiply_exact",
        lambda *a, **k: {"mode": "exact", "flop_exact": 1},
    )
    monkeypatch.setattr(
        runner.subspace, "multiply_subspace",
        lambda *a, **k: {
            "mode": "smart", "flop_exact": 2, "flop_actual": 1,
        },
    )

    runner.compare(4, Config(workdir=".", storage="auto", verbose=False))
    assert len(calls) == 1
    assert calls[0][0] == 4
    assert calls[0][2] == "auto"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
