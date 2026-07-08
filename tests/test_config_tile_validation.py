"""CPU-only tests for matmul Config.tile validation (no GPU required).

Regression guard: a non-positive ``--tile`` used to slip past Config unchecked and
then make ``gemm._tiles`` yield an empty block list, so ``multiply`` wrote no output
and returned C untouched (an all-zeros result reported as a successful run).

Run:  python tests/test_config_tile_validation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul.config import Config
from matmul.gemm import _tiles


def test_negative_tile_is_rejected():
    # A negative tile is truthy, so `cfg.tile or auto_tile(...)` would use it directly.
    try:
        Config(tile=-5)
    except ValueError as e:
        assert "tile" in str(e)
    else:
        raise AssertionError("Config(tile=-5) should raise ValueError")


def test_zero_tile_is_rejected():
    # 0 is falsy, so it would silently fall back to auto instead of honoring the request.
    try:
        Config(tile=0)
    except ValueError as e:
        assert "tile" in str(e)
    else:
        raise AssertionError("Config(tile=0) should raise ValueError")


def test_none_tile_is_allowed_for_auto():
    assert Config(tile=None).tile is None


def test_positive_tile_is_allowed():
    assert Config(tile=256).tile == 256


def test_rejected_tiles_would_have_produced_an_empty_cover():
    # Documents *why* the guard matters: the values Config now rejects are exactly the
    # ones that make the tiled path cover nothing (empty range) for any n.
    assert _tiles(1000, -5) == []


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
