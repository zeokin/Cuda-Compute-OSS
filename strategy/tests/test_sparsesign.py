"""CPU-only correctness tests for the sparse-sign sketch transform.

Drives ``SparseSignSketchTransform.basis`` with a tiny torch-CPU stand-in backend
(the real Backend needs a GPU), checking the produced basis is orthonormal, the
right shape, and actually captures col(A)/row(A)/row(B) of a low-rank couple.

Run:  python strategy/tests/test_sparsesign.py
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
    from strategy.transforms import SparseSignSketchTransform, available, get_transform


class _CPUBackend:
    """Minimal torch-CPU backend: only what Transform.basis touches."""

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
    return (rng.standard_normal((n, r)) @ rng.standard_normal((r, n))).astype(np.float32)


def test_basis_is_orthonormal_and_right_shape():
    if _skip():
        return
    n, m, r = 64, 30, 4
    A, B = _lowrank(n, r, 0), _lowrank(n, r, 1)
    Q = SparseSignSketchTransform(seed=0).basis(n, m, _CPUBackend(), np.float32, A=A, B=B)
    assert tuple(Q.shape) == (n, m)
    gram = Q.T @ Q
    assert torch.allclose(gram, torch.eye(m), atol=1e-4), "columns not orthonormal"


def test_captures_the_three_product_spaces_on_low_rank():
    if _skip():
        return
    n, m, r = 64, 30, 4          # m = 30 >> 3r = 12
    A, B = _lowrank(n, r, 0), _lowrank(n, r, 1)
    Q = SparseSignSketchTransform(seed=0).basis(n, m, _CPUBackend(), np.float32, A=A, B=B)
    Qn = Q.numpy().astype(np.float64)
    P = Qn @ Qn.T                # projector onto range(Q)
    Af, Bf = A.astype(np.float64), B.astype(np.float64)

    def rel(X, Y):
        return np.linalg.norm(X - Y) / np.linalg.norm(X)

    assert rel(Af, P @ Af) < 1e-3, "col(A) not captured"
    assert rel(Af, Af @ P) < 1e-3, "row(A) not captured"
    assert rel(Bf, Bf @ P) < 1e-3, "row(B) not captured"
    # Hence the subspace product P A P B P reconstructs A@B on this low-rank couple.
    assert rel(Af @ Bf, P @ Af @ P @ Bf @ P) < 1e-3


def test_registered_and_flops_ordering():
    if _skip():
        return
    assert "sparsesign" in available()
    assert isinstance(get_transform("sparsesign"), SparseSignSketchTransform)
    t = SparseSignSketchTransform()
    n, m = 8192, 512
    # basis cost sits between nystrom's pure QR and rsvd's dense O(2 N^2 M) sketch.
    from strategy.transforms import NystromTransform, RandomizedSVDTransform
    assert NystromTransform().basis_flops(n, m) < t.basis_flops(n, m) < \
        RandomizedSVDTransform().basis_flops(n, m)


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
