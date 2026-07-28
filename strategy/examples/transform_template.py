"""Copy-paste starting point for a new Transform. Rename the class and
`name`, fill in `basis()`, then:

    1. Local smoke test (no GPU needed):  python -m strategy.smoke
    2. Legacy reproduction (needs GPU):   python -m eval --transforms mine

See strategy/transforms.py for the full interface contract and
The square-transform API is retained for legacy research; it is not the active
attention mining lane. See the root README for current contribution scope.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy.transforms import Transform, register_transform


class MyTransform(Transform):
    """One-line description of the structure this transform exploits (e.g.
    "smoothly decaying singular values via a fixed DCT basis")."""

    name = "mine"  # rename -- this is what --transform/--transforms selects

    def basis(self, n, m, backend, dtype, A=None, B=None, frac=None):
        """Return an (n, m) array on backend.xp with ORTHONORMAL columns.

        n, m    : full dimension and target subspace dimension (m << n).
        backend : the compute backend -- use backend.xp (numpy-like: zeros,
                  arange, cos, concatenate, linalg.qr, ...), backend.matmul,
                  backend.to_device / backend.to_host. Go through backend.xp
                  rather than calling torch/numpy directly, so this code runs
                  unchanged on both the real GPU backend and the CPU
                  smoke-test shim (strategy/cpu_backend.py).
        A, B    : the two operand matrices, host-side (NumPy/memmap), if your
                  basis is DATA-DEPENDENT (like rsvd). Ignore them if your
                  basis is fixed / data-independent (like a DCT basis).
        frac    : fraction of free device memory one streamed row-block may use
                  (Config.vram_fraction when driven by the strategy; None means
                  "use the streaming default"). Forward it to any stream_gemm_*
                  helpers you call so the basis stage honours the same VRAM
                  budget as compress/reconstruct -- multiply_subspace only
                  passes frac to a basis() that declares it, so dropping the
                  parameter silently ignores --vram-fraction here.
        """
        xp = backend.xp
        # ... build an (n, m) matrix here ...
        # Q, _ = xp.linalg.qr(M)   # orthonormalize, if M isn't already orthonormal
        raise NotImplementedError("fill in your basis construction")

    def basis_flops(self, n: int, m: int) -> float:
        """FLOPs to CONSTRUCT the basis. Override this if it's non-negligible
        -- the default (0.0) means "free," which OVERSTATES your savings in
        the dominance gate if it isn't actually free. Be honest here; see
        RandomizedSVDTransform.basis_flops in strategy/transforms.py for a
        worked example."""
        return 0.0


register_transform(MyTransform.name, MyTransform)


if __name__ == "__main__":
    # Quick local check on whatever backend is available (real GPU if
    # present, else the CPU smoke-test shim) -- same check strategy.smoke
    # runs for every registered transform.
    from strategy.smoke import check_transform, _pick_backend

    backend = _pick_backend()
    try:
        ok, detail = check_transform(MyTransform.name, backend)
    except Exception as e:  # noqa: BLE001 -- mirrors strategy.smoke.main()'s per-transform handling
        ok, detail = False, f"raised {type(e).__name__}: {e}"
    print(f"{'PASS' if ok else 'FAIL'}  {MyTransform.name}: {detail}")
