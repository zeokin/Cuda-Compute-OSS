"""CPU-only tests for the FLOP-ratio phrasing (issue #213).

`flop_exact / flop_actual < 1` means the smart path did MORE work; it must not be
phrased as "Nx fewer FLOPs than exact".

Run:  python strategy/tests/test_flop_ratio_line.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy.runner import _flop_ratio_line


def test_ratio_above_one_reads_as_fewer():
    assert _flop_ratio_line(1.8) == "1.8x fewer FLOPs than exact"


def test_ratio_one_is_the_fewer_boundary():
    assert _flop_ratio_line(1.0) == "1.0x fewer FLOPs than exact"


def test_ratio_below_one_reads_as_more_not_fewer():
    s = _flop_ratio_line(0.32)          # n=128 default case
    assert "fewer" not in s
    assert "MORE FLOPs than exact" in s
    # 1 / 0.32 = 3.1x more
    assert s.startswith("3.1x MORE")


def test_small_ratio_inverts_correctly():
    assert _flop_ratio_line(0.5).startswith("2.0x MORE")


if __name__ == "__main__":
    fns = [v for kk, v in sorted(globals().items()) if kk.startswith("test_")]
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
