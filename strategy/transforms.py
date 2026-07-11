"""Pluggable subspace transforms (the "core tech" of the strategy).

A transform supplies an orthonormal N x M basis Q whose columns define the
subspace we compress into. The quality of the approximation is entirely
determined by how well Q captures the column/row spaces of A and B.

Built-in transforms: ``rsvd`` (data-dependent randomized range finder) and
``dct`` (fixed type-II DCT basis for smooth / decaying-spectrum data).
Everything else is a contribution: subclass ``Transform`` and register it.

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


class DCTTransform(Transform):
    """Fixed type-II DCT basis (lowest-frequency modes).

    Data-independent: ignores A/B. Columns are the first ``m`` orthonormal
    DCT-II frequency modes — a natural basis for smooth / decaying-spectrum
    matrices whose energy concentrates at low frequencies. Construction is a
    closed-form fill (no sketches, no QR), so ``basis_flops`` is ``O(N M)``
    rather than ``rsvd``'s ``O(N² M)``.
    """

    name = "dct"

    def basis(self, n, m, backend, dtype, A=None, B=None):
        if m < 1 or m > n:
            raise ValueError(f"dct requires 1 <= m <= n; got m={m}, n={n}")
        # Orthonormal DCT-II (ortho convention):
        #   Q[i, k] = α_k * cos(π * k * (2i + 1) / (2n))
        #   α_0 = sqrt(1/n),  α_k = sqrt(2/n) for k >= 1
        # Truncating to the first m columns preserves orthonormality because
        # the full n×n DCT-II matrix is orthogonal.
        i = np.arange(n, dtype=np.float64)[:, None]
        k = np.arange(m, dtype=np.float64)[None, :]
        Q = np.cos(np.pi * k * (2.0 * i + 1.0) / (2.0 * n))
        Q[:, 0] *= np.sqrt(1.0 / n)
        if m > 1:
            Q[:, 1:] *= np.sqrt(2.0 / n)
        return backend.to_device(Q.astype(dtype, copy=False))

    def basis_flops(self, n, m):
        # Closed-form fill of n*m entries (cos + scale). Not free — report
        # honestly so the dominance gate does not overstate savings.
        return float(n * m)


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


for _cls in (RandomizedSVDTransform, DCTTransform):
    register_transform(_cls.name, _cls)
