"""Regression tests: generate_qkv must accept the NumPy-integer seed that
AttentionSpec declares valid.

AttentionSpec type-checks its size fields -- including ``seed`` -- against
``numbers.Integral``, so ``seed=np.int64(0)`` is a *valid* spec (pinned by
tests/test_attention_spec_integral_sizes.py, whose _SIZE_FIELDS includes
"seed"). But ``torch.Generator.manual_seed`` takes a real ``int`` and raises
"TypeError: an integer is required" on an np.integer, so the only consumer of
``spec.seed`` crashed on a spec-valid value. TypeError is not in
benchmark.main's ``except (ValueError, RuntimeError, MemoryError)``, so it
escaped as an uncaught traceback instead of a clean ``error:`` + exit 2.

A numpy seed is the ordinary case: ``rng.integers(0, 1000)`` and
``np.arange(n)[i]`` both yield np.int64 -- the natural way to drive a seed sweep.

CPU-safe: skips cleanly when torch is not installed.
Run:  python tests/test_generate_qkv_integral_seed.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

if torch is not None:
    from attention.data import generate_qkv
    from attention.spec import AttentionSpec


def _skip():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


def _spec(seed):
    return AttentionSpec(batch=1, heads=1, seq=8, dim=4, window=4,
                         seed=seed, device="cpu", dtype="fp32")


def test_generate_qkv_accepts_numpy_integer_seed():
    if _skip():
        return
    for np_int in (np.int32, np.int64):
        q, k, v = generate_qkv(_spec(np_int(0)))
        assert q.shape == (1, 1, 8, 4), f"{np_int.__name__}: bad shape {q.shape}"
        assert torch.isfinite(q).all()


def test_numpy_seed_matches_plain_int_seed():
    # int() of an np.integer is exact, so the RNG stream must be identical --
    # the coercion must not change results for anyone.
    if _skip():
        return
    plain = generate_qkv(_spec(7))
    numpy_seeded = generate_qkv(_spec(np.int64(7)))
    for a, b in zip(plain, numpy_seeded):
        assert torch.equal(a, b), "numpy seed must reproduce the plain-int stream"


def test_plain_int_seed_unaffected():
    if _skip():
        return
    q, _, _ = generate_qkv(_spec(0))
    assert q.shape == (1, 1, 8, 4)


def test_seed_from_numpy_rng_works():
    # The realistic trigger: a seed drawn from a NumPy RNG is np.int64.
    if _skip():
        return
    seed = np.random.default_rng(0).integers(0, 1000)
    assert isinstance(seed, np.integer)
    q, _, _ = generate_qkv(_spec(seed))
    assert q.shape == (1, 1, 8, 4)


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
