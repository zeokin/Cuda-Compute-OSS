"""CPU-only: streamed compare rel_err must honour its host-memory budget (#293).

``block_bytes`` used to size a *single* float64 row-block while each step also
built subtraction/square temporaries on top of ``ce`` and ``cs``. A 256 MiB
budget could therefore peak near 1 GiB. The helper now charges three live
float64 blocks and reuses a diff scratch so peak matches the budget.

Run:  python strategy/tests/test_rel_err_temp_budget.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy.runner import (
    _REL_ERR_LIVE_F64_BLOCKS,
    _rel_err_row_block,
    _rel_frobenius_streamed,
)


def _naive(Ce, Cs):
    ref = np.asarray(Ce, dtype=np.float64)
    return float(
        np.linalg.norm(np.asarray(Cs, dtype=np.float64) - ref)
        / (np.linalg.norm(ref) or 1.0)
    )


def test_live_block_charge_is_three():
    assert _REL_ERR_LIVE_F64_BLOCKS == 3


def test_row_block_shrinks_by_live_factor():
    n = 4096
    row_bytes = n * 8
    # Keep both the naive and charged row counts strictly below n.
    budget = 24 * 1024**2  # 24 MiB → naive 768 rows, charged 256 rows
    naive_rows = budget // row_bytes
    charged = _rel_err_row_block(n, budget)
    assert naive_rows < n
    assert charged == naive_rows // _REL_ERR_LIVE_F64_BLOCKS
    assert charged * row_bytes * _REL_ERR_LIVE_F64_BLOCKS <= budget


def test_tiny_budget_still_clamps_to_one_row():
    assert _rel_err_row_block(512, 1) == 1


def test_streamed_result_matches_naive_after_budget_fix():
    rng = np.random.default_rng(11)
    n = 96
    Ce = rng.standard_normal((n, n)).astype(np.float32)
    Cs = Ce + 0.02 * rng.standard_normal((n, n)).astype(np.float32)
    # Force multi-block streaming under the three-block charge.
    got = _rel_frobenius_streamed(Ce, Cs, block_bytes=n * 8 * 9)
    assert abs(got - _naive(Ce, Cs)) < 1e-12


def test_block_invariant_under_tight_three_block_budget():
    rng = np.random.default_rng(12)
    n = 80
    Ce = rng.standard_normal((n, n)).astype(np.float32)
    Cs = rng.standard_normal((n, n)).astype(np.float32)
    ref = _rel_frobenius_streamed(Ce, Cs)
    for rows in (1, 3, 7, n):
        bb = rows * n * 8 * _REL_ERR_LIVE_F64_BLOCKS
        assert abs(_rel_frobenius_streamed(Ce, Cs, block_bytes=bb) - ref) < 1e-12


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
