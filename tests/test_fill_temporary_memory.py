"""Blockwise matrix fills must avoid an unnecessary second full block."""
from __future__ import annotations

import numpy as np
import pytest

from matmul import storage as matmul_storage
from strategy import storage as strategy_storage


STORAGE_MODULES = [matmul_storage, strategy_storage]


@pytest.mark.parametrize("storage", STORAGE_MODULES)
def test_random_fill_matches_single_draw_scaled_in_place(storage):
    n, seed, scale = 32, 17, 0.25
    actual = np.empty((n, n), dtype=np.float64)
    storage._fill_random(actual, seed, scale)

    expected = np.random.default_rng(seed).standard_normal((n, n))
    expected *= scale
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("storage", STORAGE_MODULES)
def test_iota_fill_matches_seeded_formula(storage):
    n, seed = 37, 11
    actual = np.empty((n, n), dtype=np.int64)
    storage._fill_iota(actual, seed)

    rng = np.random.default_rng(seed)
    row_shift = rng.integers(0, 97, size=n)
    col_shift = rng.integers(0, 97, size=n)
    expected = (
        np.arange(n)[:, None] + row_shift[:, None]
        + np.arange(n)[None, :] + col_shift[None, :]
    ) % 97
    assert np.array_equal(actual, expected)
