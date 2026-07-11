"""Pluggable subspace transforms (the "core tech" of the strategy).

A transform supplies an orthonormal N x M basis Q whose columns define the
subspace we compress into. The quality of the approximation is entirely
determined by how well Q captures the column/row spaces of A and B.

The only built-in transform is ``rsvd`` (a data-dependent randomized range
finder). Everything else is a contribution: subclass ``Transform`` and register
it.

Add your own (this is the updatable hook):

    from strategy.transforms import Transform, register_transform

    class MyTransform(Transform):
        name = "mine"
        def basis(self, n, m, backend, dtype, A=None, B=None):
            Q = ...            # (n, m) array on backend.xp, ORTHONORMAL columns
            return Q
    register_transform("mine", MyTransform)

Then select it with Config(transform="mine") or --transform mine.

Standalone: no imports from the sibling `matmul` package.
"""
from __future__ import annotations

import numpy as np


class Transform:
    """Base class. Subclasses implement ``basis`` returning an (n, m) matrix
    with orthonormal columns, living on ``backend.xp`` (GPU or CPU)."""

    name = "base"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def basis(self, n: int, m: int, backend, dtype, A=None, B=None):
        raise NotImplementedError

    def basis_flops(self, n: int, m: int) -> float:
        """FLOPs to CONSTRUCT the (n, m) basis. Added to ``multiply_subspace``'s
        reported ``flop_actual`` so the FLOP savings include basis construction --
        a mandatory, per-call, data-dependent cost that is NOT free. Override this
        when your basis is non-negligible; the default 0.0 means "negligible /
        unknown" and will OVERSTATE your savings, so report it honestly."""
        return 0.0

    @staticmethod
    def _orthonormalize(M, backend):
        Q, _ = backend.xp.linalg.qr(M)
        return Q


class RandomizedSVDTransform(Transform):
    """Data-dependent range finder over A and B (the accurate one).

    Splits the M-column budget evenly across the four spaces that must be
    captured for the product -- col(A), row(A), col(B), row(B) -- via random
    sketches, then orthonormalizes. Because all four are represented, the
    reconstruction converges to the exact product as M approaches the numerical
    rank. Sketches stream, so A/B may be disk-backed memmaps.
    """

    name = "rsvd"

    def basis(self, n, m, backend, dtype, A=None, B=None):
        if A is None or B is None:
            raise ValueError("rsvd transform needs A and B")
        from .subspace import stream_gemm_right, stream_gemm_left_t

        xp = backend.xp
        base, rem = divmod(m, 4)
        widths = [base + (1 if i < rem else 0) for i in range(4)]
        rng = np.random.default_rng(self.seed)

        def omega(w):
            return backend.to_device(
                rng.standard_normal((n, w)).astype(dtype, copy=False)
            )

        parts = []
        if widths[0]:
            parts.append(stream_gemm_right(A, omega(widths[0]), backend, dtype))
        if widths[1]:
            parts.append(stream_gemm_left_t(A, omega(widths[1]), backend, dtype))
        if widths[2]:
            parts.append(stream_gemm_right(B, omega(widths[2]), backend, dtype))
        if widths[3]:
            parts.append(stream_gemm_left_t(B, omega(widths[3]), backend, dtype))

        Y = xp.concatenate(parts, axis=1)      # (n, m)
        return self._orthonormalize(Y, backend)  # (n, m) orthonormal columns

    def basis_flops(self, n, m):
        # 4 random sketches over A and B totalling m columns cost 2*n*n*m FLOPs
        # (each width-w sketch A@Omega / A^T@Omega is 2*n*n*w, and the widths sum
        # to m), plus the QR of the (n, m) sketch ~ 2*n*m*m. Recomputed every call
        # (the sketches depend on A, B), so it is not amortizable.
        return 2.0 * n * n * m + 2.0 * n * m * m


class SubspaceIterationTransform(Transform):
    """Power-iteration ('subspace iteration') range finder -- rsvd's accurate
    sibling for data with a *decaying but not low* spectrum.

    Plain rsvd sketches each needed space (col(A), row(A), row(B)) with a single
    random projection, so at a budget M below the numerical rank the captured
    directions blur strong and weak components together. This transform runs
    ``power_iters`` steps of subspace iteration on each space -- for col(A) it
    forms ``(A Aᵀ)^q A Ω``, re-orthonormalizing between steps for stability --
    which raises the weight of the dominant singular directions by the spectral
    gap each pass. On a k^-alpha spectrum that concentrates the M budget on the
    components that actually carry the product's energy, so the reconstruction
    error at a fixed M is markedly lower than rsvd's.

    The trade-off is honest and reported: each iteration adds two streamed GEMMs
    per space, so ``basis_flops`` grows with ``power_iters`` (counted in full).
    On a genuinely full-rank spectrum there is nothing to concentrate and the
    extra passes buy nothing -- use ``rsvd`` there. Sketches stream, so A/B may
    be disk-backed memmaps.
    """

    name = "subspace-iter"
    power_iters = 1

    def basis(self, n, m, backend, dtype, A=None, B=None):
        if A is None or B is None:
            raise ValueError("subspace-iter transform needs A and B")
        from .subspace import stream_gemm_right, stream_gemm_left_t

        xp = backend.xp
        q = int(self.power_iters)
        base, rem = divmod(m, 3)
        widths = [base + (1 if i < rem else 0) for i in range(3)]
        rng = np.random.default_rng(self.seed)

        def omega(w):
            return backend.to_device(
                rng.standard_normal((n, w)).astype(dtype, copy=False)
            )

        def orth(Y):
            return self._orthonormalize(Y, backend)

        def col_range(X, w):
            """Range of X via subspace iteration: (X Xᵀ)^q X Ω, orthonormalized."""
            Y = stream_gemm_right(X, omega(w), backend, dtype)              # X Ω
            for _ in range(q):
                Y = orth(Y)                                                 # stability
                Y = stream_gemm_right(
                    X, stream_gemm_left_t(X, Y, backend, dtype), backend, dtype
                )                                                           # X (Xᵀ Y)
            return Y

        def row_range(X, w):
            """Range of Xᵀ via subspace iteration: (Xᵀ X)^q Xᵀ Ω, orthonormalized."""
            Y = stream_gemm_left_t(X, omega(w), backend, dtype)            # Xᵀ Ω
            for _ in range(q):
                Y = orth(Y)
                Y = stream_gemm_left_t(
                    X, stream_gemm_right(X, Y, backend, dtype), backend, dtype
                )                                                           # Xᵀ (X Y)
            return Y

        parts = []
        if widths[0]:
            parts.append(col_range(A, widths[0]))   # col(A)
        if widths[1]:
            parts.append(row_range(A, widths[1]))   # row(A)
        if widths[2]:
            parts.append(row_range(B, widths[2]))   # row(B)

        Y = xp.concatenate(parts, axis=1)                                   # (n, m)
        return self._orthonormalize(Y, backend)

    def basis_flops(self, n, m):
        # Initial sketches total width m -> 2 n^2 m. Each of the q iterations does
        # two streamed GEMMs per space (X and Xᵀ) over total width m -> 4 n^2 m,
        # plus a re-orthonormalizing QR of the (n, m) block ~ 2 n m^2; a final QR
        # adds one more 2 n m^2. Counted in full so the FLOP savings stay honest.
        q = int(self.power_iters)
        return (2.0 + 4.0 * q) * n * n * m + (2.0 * q + 2.0) * n * m * m


_REGISTRY: dict[str, type[Transform]] = {}


def register_transform(name: str, cls: type[Transform]) -> None:
    _REGISTRY[name] = cls


def get_transform(name_or_instance, seed: int = 0) -> Transform:
    if isinstance(name_or_instance, Transform):
        return name_or_instance
    if name_or_instance not in _REGISTRY:
        raise KeyError(
            f"unknown transform {name_or_instance!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name_or_instance](seed=seed)


def available() -> list[str]:
    return sorted(_REGISTRY)


for _cls in (RandomizedSVDTransform, SubspaceIterationTransform):
    register_transform(_cls.name, _cls)
