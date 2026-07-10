"""CPU tests for eval JSON transparency scoring (Fixes #93).

Run:  python tests/test_eval_perf_score.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics


def test_transparency_score_stays_ungated_below_floor():
    acc = 0.002
    peak = 1e6
    lat = 0.5
    raw = metrics.transparency_score(acc, peak, lat, "mib")
    gated = metrics.score(acc, peak, lat, 0.8, "mib")
    assert raw > 0.0
    assert gated == 0.0


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
