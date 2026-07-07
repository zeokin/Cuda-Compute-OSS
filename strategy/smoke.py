"""Tier-1 (local, informational, never binding) smoke test for every
registered Transform -- see docs/testing-strategy.md. Tiny N, CPU-friendly,
finishes in seconds: checks shape, orthonormality, and no NaN/Inf. This is a
sanity pre-flight, not a score -- the real scorecard is `python -m eval` on
a real GPU, which is what actually decides a PR's verdict.

    python -m strategy.smoke
    python -m strategy.smoke rsvd
"""
from __future__ import annotations

import sys

import numpy as np

from .cpu_backend import CPUBackend
from .storage import generate
from .transforms import available, get_transform

N = 64
M = 16
DTYPE = np.float32


def _pick_backend():
    """Prefer the real GPU backend when one exists (more representative of
    the real scoring path); fall back to the CPU shim so this always runs,
    on any miner's machine, GPU or not."""
    try:
        from .backend import Backend
        return Backend(device=0, verbose=False)
    except Exception:
        return CPUBackend(device=0, verbose=False)


def check_transform(name: str, backend) -> tuple:
    """Return (ok, detail) for one registered transform's basis()."""
    transform = get_transform(name, seed=0)
    A = generate(N, DTYPE, False, None, seed=1, fill="random")
    B = generate(N, DTYPE, False, None, seed=2, fill="random")

    Q = transform.basis(N, M, backend, DTYPE, A=A, B=B)

    shape = tuple(Q.shape)
    if shape != (N, M):
        return False, f"basis() returned shape {shape}, expected {(N, M)}"

    Qh = backend.to_host(Q).astype(np.float64)
    if not np.isfinite(Qh).all():
        return False, "basis() contains NaN/Inf"

    gram = Qh.T @ Qh
    err = float(np.linalg.norm(gram - np.eye(M)))
    if err > 1e-2:
        return False, f"columns are not orthonormal (||QtQ - I|| = {err:.2e})"

    return True, f"OK (||QtQ - I|| = {err:.2e})"


def pick_transforms(argv=None) -> list[str]:
    """Return the transform names requested on the command line.

    With no names, smoke-check every registered transform. With explicit names,
    keep their input order and fail fast on unknown entries so a contributor
    does not accidentally think they tested something they did not.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    names = available()
    if not args:
        return names

    known = set(names)
    missing = [name for name in args if name not in known]
    if missing:
        raise KeyError(
            f"unknown transform(s) {missing}; available: {sorted(names)}"
        )
    return args


def main(argv=None) -> int:
    backend = _pick_backend()
    print(f"[smoke] backend: {backend.name}  (N={N}, M={M}, dtype={np.dtype(DTYPE).name})")

    names = pick_transforms(argv)
    if not names:
        print("[smoke] no transforms registered -- nothing to check")
        return 0

    failed = 0
    for name in names:
        try:
            ok, detail = check_transform(name, backend)
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"raised {type(e).__name__}: {e}"
        print(f"[smoke] {'PASS' if ok else 'FAIL'}  {name:<12} {detail}")
        if not ok:
            failed += 1

    print(f"\n[smoke] {len(names) - failed}/{len(names)} transforms passed "
          f"(informational only -- this is not a scorecard)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
