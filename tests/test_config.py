"""CPU unit tests for matmul.Config argument validation.

Pure dataclass checking -- no GPU/PyTorch needed, so these run anywhere (unlike
tests/test_correctness.py, which drives the GPU engine and skips without a
device).

Run:  python tests/test_config.py        (or)   python -m pytest tests/test_config.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul.config import Config


def _rejects(**kw) -> bool:
    try:
        Config(verbose=False, **kw)
        return False
    except ValueError:
        return True


def test_tile_none_means_auto():
    assert Config(tile=None, verbose=False).tile is None


def test_positive_tile_ok():
    assert Config(tile=128, verbose=False).tile == 128


def test_nonpositive_tile_rejected():
    # A non-positive tile makes gemm._tiles() yield an empty schedule, so the
    # tiled path never writes C and returns the uninitialised output buffer as
    # the "exact" product -- a silent wrong answer. Config must reject it up
    # front, like it already does for dtype / vram_fraction / storage.
    for bad in (0, -1, -128):
        assert _rejects(tile=bad), f"tile={bad} should be rejected"


def test_existing_validations_still_hold():
    # regression: the other knobs stay guarded.
    assert _rejects(dtype="fp8")
    assert _rejects(vram_fraction=0.0)
    assert _rejects(vram_fraction=1.5)
    assert _rejects(storage="cloud")


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
