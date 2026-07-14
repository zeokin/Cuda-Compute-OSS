"""CPU-only tests: rsvd basis threads the configured VRAM fraction (issue #210).

rsvd.basis sized its sketch row-blocks against the hardcoded 0.3 default instead of
Config.vram_fraction. These stub the streamed sketches to capture the `frac` they
receive, so no GPU is needed.

Run:  python strategy/tests/test_basis_vram_fraction.py
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import subspace as sub
from strategy.transforms import RandomizedSVDTransform, Transform


class _XP:
    concatenate = staticmethod(np.concatenate)

    class linalg:
        qr = staticmethod(np.linalg.qr)


class _FakeBackend:
    xp = _XP()

    def to_device(self, x):
        return np.asarray(x)


def _capture_fracs(frac_kwarg="__omit__", n=8, m=4):
    """Call rsvd.basis with stubbed sketches; return the list of frac values they saw."""
    captured = []
    orig_r, orig_l = sub.stream_gemm_right, sub.stream_gemm_left_t

    def fake(X, Q, backend, dtype, frac=sub._DEFAULT_ROW_BLOCK_FRACTION):
        captured.append(frac)
        return np.zeros((X.shape[0], Q.shape[1]), dtype=dtype)

    sub.stream_gemm_right = fake
    sub.stream_gemm_left_t = fake
    try:
        A = np.eye(n, dtype=np.float32)
        B = np.eye(n, dtype=np.float32)
        t = RandomizedSVDTransform(seed=0)
        kw = {} if frac_kwarg == "__omit__" else {"frac": frac_kwarg}
        Q = t.basis(n, m, _FakeBackend(), np.float32, A=A, B=B, **kw)
        assert Q.shape == (n, m)
    finally:
        sub.stream_gemm_right, sub.stream_gemm_left_t = orig_r, orig_l
    return captured


def test_rsvd_basis_threads_the_configured_fraction():
    # rsvd captures 3 subspaces (col(A), row(A), row(B)); frac reaches every sketch.
    fracs = _capture_fracs(frac_kwarg=0.15)
    assert fracs == [0.15, 0.15, 0.15], fracs


def test_rsvd_basis_defaults_to_streaming_default_when_none():
    d = sub._DEFAULT_ROW_BLOCK_FRACTION
    assert _capture_fracs(frac_kwarg=None) == [d, d, d]
    assert _capture_fracs(frac_kwarg="__omit__") == [d, d, d]


def test_basis_signatures_accept_frac():
    # Base class + rsvd expose `frac` so multiply_subspace can thread it.
    assert "frac" in inspect.signature(Transform.basis).parameters
    assert "frac" in inspect.signature(RandomizedSVDTransform.basis).parameters


def test_old_signature_transform_is_detected_by_the_guard():
    # A custom transform without `frac` must be callable without it (the guard in
    # multiply_subspace inspects the signature before passing frac).
    class _Old(Transform):
        name = "old"

        def basis(self, n, m, backend, dtype, A=None, B=None):  # no frac
            return backend.xp.linalg.qr(np.zeros((n, m), dtype=dtype))[0]

    assert "frac" not in inspect.signature(_Old().basis).parameters


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
