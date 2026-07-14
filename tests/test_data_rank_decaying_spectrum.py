"""`--data-rank` controls the rank of BOTH the lowrank and decaying-spectrum
fills (storage.generate uses it for each), and its default is max(1, n//32) --
not the bare n//32 the CLI help used to claim. These pure-NumPy checks pin that
contract, backing the corrected --data-rank help text.
"""
from __future__ import annotations

import numpy as np

from strategy import storage


def test_data_rank_sets_decaying_spectrum_rank():
    for r in (2, 8, 20):
        m = storage.generate(64, np.float64, False, None, seed=0,
                              fill="decaying-spectrum", data_rank=r)
        assert np.linalg.matrix_rank(m) == r


def test_data_rank_sets_lowrank_rank():
    for r in (3, 10):
        m = storage.generate(64, np.float64, False, None, seed=0,
                              fill="lowrank", data_rank=r)
        assert np.linalg.matrix_rank(m) == r


def test_default_data_rank_is_floored_at_one():
    # n // 32 == 0 for small n, but the default is max(1, n//32), so a
    # decaying-spectrum matrix is at least rank 1 (never rank 0 / all-zero).
    m = storage.generate(16, np.float64, False, None, seed=0,
                         fill="decaying-spectrum", data_rank=None)
    assert np.linalg.matrix_rank(m) == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
