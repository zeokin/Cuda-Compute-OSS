"""CPU-safe tests for strategy.smoke's transform selection logic."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import smoke


def test_pick_transforms_defaults_to_all_registered():
    names = smoke.pick_transforms([])
    assert names == ["rsvd"]


def test_pick_transforms_preserves_requested_order():
    names = smoke.pick_transforms(["rsvd"])
    assert names == ["rsvd"]


def test_pick_transforms_rejects_unknown_names():
    try:
        smoke.pick_transforms(["missing-transform"])
    except KeyError as e:
        assert "missing-transform" in str(e)
    else:
        raise AssertionError("pick_transforms should reject unknown names")


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
