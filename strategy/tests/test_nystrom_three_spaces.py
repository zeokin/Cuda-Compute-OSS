"""CPU-only tests: nystrom needs only 3 landmark spaces, not 4.

col(B) is redundant (Ĉ = P A P B P = A B once range(Q) ⊇ col(A),row(A),row(B)),
so splitting M across 3 spaces gives ~M/3 columns each and recovers a rank-r
product at M ≳ 3r — where the old 4-space split (~M/4 each) would still be starved.

Drives ``NystromTransform.basis`` with a tiny torch-CPU stand-in backend.
Run:  python strategy/tests/test_nystrom_three_spaces.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

if torch is not None:
    from strategy.transforms import NystromTransform


class _CPUBackend:
    class _XP:
        @staticmethod
        def concatenate(tensors, axis=0):
            return torch.cat(list(tensors), dim=axis)

        class linalg:
            @staticmethod
            def qr(m):
                return torch.linalg.qr(m)

    xp = _XP()

    def to_device(self, x):
        return torch.as_tensor(np.ascontiguousarray(x))


def _skip():
    if torch is None:
        print("SKIP  torch not installed")
        return True
    return False


def _lowrank(n, r, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, r)) @ rng.standard_normal((r, n))).astype(np.float64)


def _basis(n, m):
    A, B = _lowrank(n, 4, 0), _lowrank(n, 4, 1)
    Q = NystromTransform(seed=0).basis(n, m, _CPUBackend(), np.float64, A=A, B=B)
    return A, B, Q.numpy()


def test_recovers_rank_r_below_the_old_4r_threshold():
    # r=4: M=15 gives ~5 columns per space with 3 spaces (spans rank 4), but the
    # old 4-space split gave M/4 = 3 < r = 4 per space and could NOT recover here.
    if _skip():
        return
    n, m = 64, 15
    A, B, Q = _basis(n, m)
    P = Q @ Q.T

    def rel(X, Y):
        return np.linalg.norm(X - Y) / np.linalg.norm(X)

    assert rel(A @ B, P @ A @ P @ B @ P) < 1e-6, "did not recover A@B at M=3r+"
    assert rel(A, P @ A) < 1e-6 and rel(A, A @ P) < 1e-6, "col(A)/row(A) not captured"
    assert rel(B, B @ P) < 1e-6, "row(B) not captured"


def test_basis_is_orthonormal_and_shaped():
    if _skip():
        return
    n, m = 64, 15
    _, _, Q = _basis(n, m)
    assert Q.shape == (n, m)
    assert np.allclose(Q.T @ Q, np.eye(m), atol=1e-8)


def test_three_way_split_of_m():
    # 3 spaces, so M columns split ~M/3 each (not M/4). At M=15 that's [5,5,5].
    if _skip():
        return
    n, m = 64, 15
    _, _, Q = _basis(n, m)
    assert Q.shape[1] == m  # all M columns used across exactly the 3 needed spaces


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
