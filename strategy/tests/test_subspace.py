"""Tests for the standalone subspace strategy and its transforms (GPU/PyTorch).

CCO computes on the GPU only, so these need a CUDA/MPS device; they skip cleanly
when none is present.

Run:  python strategy/tests/test_subspace.py   (or via pytest)
"""
import os
import sys

import numpy as np

# make the project root importable so `import strategy` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import subspace_matmul, Config, register_transform, available
from strategy.backend import Backend
from strategy import subspace
from strategy.transforms import Transform, get_transform


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()
                    or (getattr(torch.backends, "mps", None)
                        and torch.backends.mps.is_available()))
    except Exception:  # noqa: BLE001
        return False


HAVE_GPU = _gpu_available()
BK = Backend(verbose=False) if HAVE_GPU else None

# Under pytest, skip the whole module cleanly when no GPU is present — every test
# drives the GPU backend through BK, which is None without a device. (The
# __main__ runner does its own skip for direct `python .../test_subspace.py`.)
try:
    import pytest
    pytestmark = pytest.mark.skipif(
        not HAVE_GPU, reason="no CUDA/MPS GPU; CCO computes on GPU only")
except ImportError:
    pass


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0))


def _run(A, B, m, transform, dtype="fp64"):
    cfg = Config(dtype=dtype, rank_m=m, transform=transform, verbose=False)
    C = np.zeros_like(A)
    subspace.multiply_subspace(A, B, C, BK, cfg)
    return C


# -- flop accounting (pure arithmetic, no GPU needed) ---------------------
def test_flop_actual_matches_matmul_sum():
    # Cross-check subspace._flop_actual against an independently-summed FLOP
    # count for every matmul multiply_subspace actually performs: two
    # compress() calls (X@Q then Q.T@(X@Q) each), the (m,m) core product,
    # and reconstruct() (Q@Ctil then (...)@Q.T).
    for n, m in [(8, 2), (12000, 1500), (24, 24), (100, 1)]:
        compress_call = 2 * n * n * m + 2 * n * m * m
        expected = (2 * compress_call            # compress(A) + compress(B)
                    + 2 * m * m * m               # Atil @ Btil
                    + 2 * n * m * m + 2 * n * n * m)  # reconstruct
        assert subspace._flop_actual(n, m) == expected


def test_basis_flops_counted_in_reported_flop_actual():
    # The transform's basis construction (rsvd: sketches + QR) is a mandatory
    # O(n^2 m) per-call cost. multiply_subspace reports flop_actual = core
    # (_flop_actual) PLUS transform.basis_flops, so the "FLOPs saved" figure does
    # not overstate the win. A bare Transform reports 0 by default.
    from strategy.transforms import RandomizedSVDTransform, Transform
    n, m = 4096, 512
    rsvd = RandomizedSVDTransform()
    assert rsvd.basis_flops(n, m) == 2.0 * n * n * m + 2.0 * n * m * m
    assert rsvd.basis_flops(n, m) > 0.0

    class _Bare(Transform):
        name = "_bare"
    assert _Bare().basis_flops(n, m) == 0.0

    # reported flop_actual (core + basis) strictly exceeds the core-only count.
    assert subspace._flop_actual(n, m) + rsvd.basis_flops(n, m) > subspace._flop_actual(n, m)


# -- streaming primitives -------------------------------------------------
# The primitives stream the rows of X from the host but expect the *resident*
# operand (Q / Om / Ctil) to already be a device tensor, exactly as production
# transforms hand them one via backend.to_device. Pass host NumPy and torch.matmul
# raises; likewise the device-resident result must come back through to_host
# before it can be compared with NumPy.
def test_compress_matches_direct():
    rng = np.random.default_rng(0)
    n, m = 40, 12
    X = rng.standard_normal((n, n))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    Qd = BK.to_device(Q.astype(np.float64))
    got = BK.to_host(subspace.compress(X, Qd, BK, np.float64))
    assert _rel(got, Q.T @ X @ Q) < 1e-12


def test_reconstruct_matches_direct():
    rng = np.random.default_rng(1)
    n, m = 40, 12
    Ctil = rng.standard_normal((m, m))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    Qd = BK.to_device(Q.astype(np.float64))
    Ctild = BK.to_device(Ctil.astype(np.float64))
    C = np.zeros((n, n))
    subspace.reconstruct(Ctild, Qd, C, BK, np.float64)
    assert _rel(C, Q @ Ctil @ Q.T) < 1e-12


def test_stream_primitives():
    rng = np.random.default_rng(2)
    n, m = 32, 7
    X = rng.standard_normal((n, n))
    Om = rng.standard_normal((n, m))
    Omd = BK.to_device(Om.astype(np.float64))
    right = BK.to_host(subspace.stream_gemm_right(X, Omd, BK, np.float64))
    left = BK.to_host(subspace.stream_gemm_left_t(X, Omd, BK, np.float64))
    assert _rel(right, X @ Om) < 1e-12
    assert _rel(left, X.T @ Om) < 1e-12


def test_exact_baseline_matches_numpy():
    rng = np.random.default_rng(3)
    n = 48
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    C = np.zeros((n, n))
    subspace.multiply_exact(A, B, C, BK, Config(dtype="fp64", verbose=False))
    assert _rel(C, A @ B) < 1e-12


def test_exact_baseline_memmap_matches_numpy():
    """Disk-backed inputs must stream both operands, not upload full B to GPU."""
    import shutil
    import tempfile

    from strategy import storage

    workdir = tempfile.mkdtemp(prefix="cco_exact_memmap_")
    try:
        n = 64
        cfg = Config(dtype="fp64", verbose=False, workdir=workdir)
        pa = os.path.join(workdir, "A.dat")
        pb = os.path.join(workdir, "B.dat")
        pc = os.path.join(workdir, "C.dat")
        A = storage.generate(n, np.float64, True, pa, 0, "iota")
        B = storage.generate(n, np.float64, True, pb, 1, "iota")
        C = storage.allocate(n, np.float64, True, pc)
        info = subspace.multiply_exact(A, B, C, BK, cfg)
        assert info["mode"] == "exact(streamed)"
        ref = A.astype(np.float64) @ B.astype(np.float64)
        assert _rel(C, ref) < 1e-12
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# -- exactness / accuracy properties --------------------------------------
def test_exact_when_m_equals_n():
    # M = N => Q spans R^N => P = I => exact. rsvd's four sketches concatenated
    # to N columns span the full space for full-rank A/B.
    rng = np.random.default_rng(4)
    n = 24
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    assert _rel(_run(A, B, n, "rsvd"), A @ B) < 1e-9


def test_custom_transform_exact_with_shared_basis():
    # A, B share an eigenbasis Q0 (rank r); a transform that returns Q0 with
    # M = r reconstructs exactly — the mechanism a contributor plugs into.
    rng = np.random.default_rng(5)
    n, r = 50, 6
    Q0 = np.linalg.qr(rng.standard_normal((n, r)))[0]
    A = Q0 @ np.diag(rng.standard_normal(r)) @ Q0.T
    B = Q0 @ np.diag(rng.standard_normal(r)) @ Q0.T

    class Shared(Transform):
        name = "shared"
        def basis(self, n, m, backend, dtype, A=None, B=None):
            return backend.to_device(Q0.astype(dtype, copy=False))

    assert _rel(_run(A, B, r, Shared()), A @ B) < 1e-9


def test_rsvd_recovers_low_rank_product():
    rng = np.random.default_rng(6)
    n, r, m = 96, 5, 48                     # m/4 = 12 >= r per captured space
    A = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
    B = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
    assert _rel(_run(A, B, m, "rsvd"), A @ B) < 1e-8


def test_rsvd_more_dims_not_worse_on_low_rank():
    # For low-rank data, a larger M can only capture at least as much.
    rng = np.random.default_rng(7)
    n, r = 60, 8
    A = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
    B = rng.standard_normal((n, r)) @ rng.standard_normal((r, n))
    assert _rel(_run(A, B, 48, "rsvd"), A @ B) <= _rel(_run(A, B, 16, "rsvd"), A @ B) + 1e-9


# -- registry / extensibility --------------------------------------------
def test_registry_custom_transform():
    class IdentityBlock(Transform):
        name = "idblock"

        def basis(self, n, m, backend, dtype, A=None, B=None):
            Q = np.zeros((n, m), dtype=dtype)
            Q[np.arange(m), np.arange(m)] = 1.0
            return backend.to_device(Q)

    register_transform("idblock", IdentityBlock)
    assert "idblock" in available()
    assert isinstance(get_transform("idblock"), IdentityBlock)
    rng = np.random.default_rng(8)
    n = 20
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    assert _rel(_run(A, B, n, "idblock"), A @ B) < 1e-9   # M=N => identity => exact


def test_public_api():
    rng = np.random.default_rng(9)
    n, r = 64, 8
    A = (rng.standard_normal((n, r)) @ rng.standard_normal((r, n))).astype(np.float32)
    B = (rng.standard_normal((n, r)) @ rng.standard_normal((r, n))).astype(np.float32)
    C = subspace_matmul(A, B, config=Config(transform="rsvd", rank_m=40,
                                            verbose=False))
    assert _rel(C, A @ B) < 1e-3


if __name__ == "__main__":
    if not HAVE_GPU:
        print("SKIP  all tests (no CUDA/MPS GPU; CCO computes on GPU only)")
        sys.exit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
