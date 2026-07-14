"""Tests for the pinned per-track regime registry (eval.tracks).

Pure data + lookups, no GPU. These pin the regime each track scores at so a
verdict can't drift when a submission picks a different rank/M.

Run:  python eval/tests/test_tracks.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval import tracks
from eval.evaluator import EvalConfig, default_floor


def test_pinned_regimes():
    lr = tracks.get_track("low-rank")
    assert (lr.fill, lr.data_rank, lr.rank_m, lr.accuracy_floor) == ("lowrank", 16, 64, 0.95)
    fr = tracks.get_track("full-rank")
    assert (fr.fill, fr.data_rank, fr.rank_m, fr.accuracy_floor) == ("random", None, None, 0.80)
    ds = tracks.get_track("decaying-spectrum")
    assert (ds.fill, ds.data_rank, ds.rank_m, ds.accuracy_floor) == ("decaying-spectrum", 64, 256, 0.90)


def test_low_rank_pin_is_inclusive_M64():
    # M=64 = 4r for rank-16: the M where BOTH 3-way (needs 3r=48) and 4-way
    # (needs 4r=64) methods fully recover, so the pin isn't rigged to one basis.
    lr = tracks.get_track("low-rank")
    assert lr.rank_m == 4 * lr.data_rank


def test_track_for_fill():
    assert tracks.track_for_fill("random") == "full-rank"
    assert tracks.track_for_fill("iota") == "full-rank"
    assert tracks.track_for_fill("lowrank") == "low-rank"
    assert tracks.track_for_fill("decaying-spectrum") == "decaying-spectrum"
    assert tracks.track_for_fill("something-new") == "full-rank"   # safe default


def test_track_floor_matches_specs():
    assert tracks.track_floor("random") == 0.80
    assert tracks.track_floor("lowrank") == 0.95
    assert tracks.track_floor("decaying-spectrum") == 0.90


def test_get_track_rejects_unknown():
    try:
        tracks.get_track("no-such-track")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for an unknown track")


def test_evaluator_floor_comes_from_tracks():
    # EvalConfig (floor=None) resolves per-track via the SAME source (tracks),
    # so the scorer and the registry can never disagree.
    assert EvalConfig(fill="lowrank").accuracy_floor == tracks.track_floor("lowrank")
    assert default_floor("decaying-spectrum") == tracks.track_floor("decaying-spectrum")


def test_accuracy_floors_dict():
    assert tracks.accuracy_floors() == {
        "full-rank": 0.80, "low-rank": 0.95, "decaying-spectrum": 0.90,
    }


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
