"""Public configuration knobs that become sizes must reject non-integers early."""
from __future__ import annotations

import math

import numpy as np
import pytest

from matmul.config import Config as MatmulConfig
from strategy.config import Config as StrategyConfig

_BOTH_CONFIGS = pytest.mark.parametrize(
    "Config", [MatmulConfig, StrategyConfig], ids=["matmul", "strategy"]
)


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


# vram_fraction is the fraction of free VRAM a tile/row-block may occupy. It is
# the one numeric knob that used to skip the isinstance guard the others carry,
# so a None / non-numeric value raised a raw ``TypeError`` from the ``<`` range
# check instead of a clean ValueError.
@_BOTH_CONFIGS
@pytest.mark.parametrize("bad", [None, "0.6", True, math.nan, math.inf])
def test_config_rejects_non_numeric_vram_fraction(Config, bad):
    with pytest.raises(ValueError, match="vram_fraction must be a number"):
        Config(vram_fraction=bad)


@_BOTH_CONFIGS
@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 0.96])
def test_config_rejects_out_of_range_vram_fraction(Config, bad):
    with pytest.raises(ValueError, match="vram_fraction must be a number"):
        Config(vram_fraction=bad)


@_BOTH_CONFIGS
def test_config_accepts_valid_vram_fraction(Config):
    assert Config(vram_fraction=0.6).vram_fraction == 0.6
    assert Config(vram_fraction=0.95).vram_fraction == 0.95
    # NumPy float scalars are common when a value comes from array metadata.
    assert float(Config(vram_fraction=np.float64(0.5)).vram_fraction) == 0.5


# workdir is joined into memmap paths on the disk-backed path; a non-str value
# passes construction and only blows up later inside os.path, mid-run.
@_BOTH_CONFIGS
@pytest.mark.parametrize("bad", [None, 123, b"dir"])
def test_config_rejects_non_string_workdir(Config, bad):
    with pytest.raises(ValueError, match="workdir must be a string"):
        Config(workdir=bad)


@_BOTH_CONFIGS
def test_config_accepts_string_workdir(Config):
    assert Config(workdir="./somedir").workdir == "./somedir"
