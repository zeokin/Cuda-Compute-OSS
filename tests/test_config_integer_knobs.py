"""Public configuration knobs that become sizes must reject non-integers early."""
from __future__ import annotations

import math

import numpy as np
import pytest

from matmul.config import Config as MatmulConfig
from strategy.config import Config as StrategyConfig


@pytest.mark.parametrize("tile", [1.5, True, "128"])
def test_matmul_config_rejects_non_integer_tile(tile):
    with pytest.raises(ValueError, match="tile must be a positive integer"):
        MatmulConfig(tile=tile)


@pytest.mark.parametrize("rank_m", [1.5, True, "8"])
def test_strategy_config_rejects_non_integer_rank_m(rank_m):
    with pytest.raises(ValueError, match="rank_m must be an integer"):
        StrategyConfig(rank_m=rank_m)


def test_integer_like_sizes_remain_supported():
    # NumPy integer scalars are common when a size comes from array metadata.
    assert MatmulConfig(tile=np.int64(128)).tile == 128
    assert StrategyConfig(rank_m=np.int64(8)).rank_m == 8


@pytest.mark.parametrize("Config", [MatmulConfig, StrategyConfig])
@pytest.mark.parametrize("value", [None, "0.6", True, math.nan, math.inf])
def test_config_rejects_invalid_vram_fraction_cleanly(Config, value):
    with pytest.raises(ValueError, match="vram_fraction must be a finite number"):
        Config(vram_fraction=value)


@pytest.mark.parametrize("Config", [MatmulConfig, StrategyConfig])
@pytest.mark.parametrize("value", [None, 123, b"directory"])
def test_config_rejects_non_string_workdir(Config, value):
    with pytest.raises(ValueError, match="workdir must be a string"):
        Config(workdir=value)
