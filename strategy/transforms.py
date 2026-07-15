"""Pluggable subspace transforms (the "core tech" of the strategy).

A transform supplies an orthonormal N x M basis Q whose columns define the
subspace we compress into. The quality of the approximation is entirely
determined by how well Q captures the column/row spaces of A and B.

Built-in transforms: ``rsvd`` (data-dependent randomized range finder),
``nystrom`` (landmark column sampling for low-rank data), and ``power-rsvd``
(``rsvd`` plus q subspace-iteration steps for decaying spectra). Everything else
is a contribution: subclass ``Transform`` and register it.

Add your own (this is the updatable hook):

    from strategy.transforms import Transform, register_transform

    class MyTransform(Transform):
        name = "mine"
        def basis(self, n, m, backend, dtype, A=None, B=None, frac=None):
            Q = ...            # (n, m) array on backend.xp, ORTHONORMAL columns
            return Q           # pass frac to any streamed stream_gemm_* helpers
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

    def basis(self, n: int, m: int, backend, dtype, A=None, B=None, frac=None):
        """Return an (n, m) orthonormal basis. ``frac`` is the fraction of free
        device memory a streamed row-block may use (``Config.vram_fraction`` when
        driven by the strategy); forward it to any ``stream_gemm_*`` helpers so the
        basis stage honours the same VRAM budget as compress/reconstruct. ``None``
        means "use the streaming default"."""
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

    The reconstruction is ``Ĉ = P A P B P`` with the projector ``P = Q Qᵀ``, so
    ``Ĉ = A @ B`` exactly once range(Q) contains col(A), row(A), row(B):

        P A = A   needs  col(A) ⊆ range(Q)
        A P = A   needs  row(A) ⊆ range(Q)
        B P = B   needs  row(B) ⊆ range(Q)   (then P A P B P = A B P = A B)

    Three spaces are necessary and sufficient -- col(B) is redundant. We split
    the M-column budget across those three via random sketches, then
    orthonormalize; exact recovery of a rank-r product needs only ``M ≳ 3r``
    instead of ``4r``. Sketches stream, so A/B may be disk-backed memmaps.
    """

    name = "rsvd"

    def basis(self, n, m, backend, dtype, A=None, B=None, frac=None):
        if A is None or B is None:
            raise ValueError("rsvd transform needs A and B")
        from .subspace import (
            _DEFAULT_ROW_BLOCK_FRACTION,
            stream_gemm_left_t,
            stream_gemm_right,
        )

        # Honour the strategy's VRAM budget (Config.vram_fraction) for the sketch
        # row-blocks, like compress/reconstruct do -- otherwise the basis stage
        # silently uses the 0.3 default and can OOM at a low --vram-fraction.
        if frac is None:
            frac = _DEFAULT_ROW_BLOCK_FRACTION

        xp = backend.xp
        base, rem = divmod(m, 3)
        widths = [base + (1 if i < rem else 0) for i in range(3)]
        rng = np.random.default_rng(self.seed)

        def omega(w):
            return backend.to_device(
                rng.standard_normal((n, w)).astype(dtype, copy=False)
            )

        parts = []
        if widths[0]:
            parts.append(stream_gemm_right(A, omega(widths[0]), backend, dtype, frac))   # col(A): A Ω
        if widths[1]:
            parts.append(stream_gemm_left_t(A, omega(widths[1]), backend, dtype, frac))  # row(A): Aᵀ Ω
        if widths[2]:
            parts.append(stream_gemm_left_t(B, omega(widths[2]), backend, dtype, frac))  # row(B): Bᵀ Ω

        Y = xp.concatenate(parts, axis=1)      # (n, m)
        return self._orthonormalize(Y, backend)  # (n, m) orthonormal columns

    def basis_flops(self, n, m):
        # 3 random sketches over A and B totalling m columns cost 2*n*n*m FLOPs
        # (each width-w sketch A@Omega / A^T@Omega is 2*n*n*w, and the widths sum
        # to m), plus the QR of the (n, m) sketch ~ 2*n*m*m. Recomputed every call
        # (the sketches depend on A, B), so it is not amortizable.
        return 2.0 * n * n * m + 2.0 * n * m * m


class NystromTransform(Transform):
    """Landmark / Nyström column sampling over A and B.

    Splits the M-column budget across col(A), row(A), col(B), and row(B) —
    the same four spaces ``rsvd`` sketches — but forms each block by gathering
    random landmark columns (or rows-as-columns) instead of random projections.
    On genuine low-rank couples the landmarks span those spaces once enough
    columns are drawn, so the thin QR that follows is enough; basis cost is
    essentially the QR (``~2 N M²``), not ``rsvd``'s ``~2 N² M`` sketches.
    """

    name = "nystrom"

    def basis(self, n, m, backend, dtype, A=None, B=None):
        if A is None or B is None:
            raise ValueError("nystrom transform needs A and B")
        if m < 1 or m > n:
            raise ValueError(f"nystrom requires 1 <= m <= n; got m={m}, n={n}")

        base, rem = divmod(m, 4)
        widths = [base + (1 if i < rem else 0) for i in range(4)]
        rng = np.random.default_rng(self.seed)

        def landmark_cols(X, w):
            # Gather w distinct columns of X into an (n, w) host block.
            idx = rng.choice(n, size=w, replace=False)
            return np.asarray(X[:, idx]).astype(dtype, copy=False)

        def landmark_rows_as_cols(X, w):
            # Rows of X as columns of Xᵀ — captures the row space.
            idx = rng.choice(n, size=w, replace=False)
            return np.asarray(X[idx, :]).T.astype(dtype, copy=False)

        parts = []
        if widths[0]:
            parts.append(backend.to_device(landmark_cols(A, widths[0])))
        if widths[1]:
            parts.append(backend.to_device(landmark_rows_as_cols(A, widths[1])))
        if widths[2]:
            parts.append(backend.to_device(landmark_cols(B, widths[2])))
        if widths[3]:
            parts.append(backend.to_device(landmark_rows_as_cols(B, widths[3])))

        Y = backend.xp.concatenate(parts, axis=1)  # (n, m)
        return self._orthonormalize(Y, backend)

    def basis_flops(self, n, m):
        # Column/row gathers are memory traffic, not FLOPs. The mandatory cost
        # is the thin QR of the (n, m) landmark stack (~2 n m²).
        return 2.0 * n * m * m


class PowerIterationTransform(Transform):
    """``rsvd`` range finder + ``q`` steps of randomized subspace iteration.

    ``rsvd`` takes a single sketch of each space (``AΩ``, ``AᵀΩ``, ``BᵀΩ``) and
    orthonormalizes. When the spectrum only *decays* (``σ_k ~ k^-α``) instead of
    being exactly low-rank, that single sketch leaks tail energy into ``Q`` and the
    approximation ``Ĉ = P A P B P`` struggles to clear the accuracy floor. This
    transform sharpens each sketch with ``q`` re-orthonormalized subspace-iteration
    steps (Halko–Martinsson–Tropp 2011, Alg. 4.4):

        col(A):  Y = A Ω;   repeat q× :  Y = orth( A (Aᵀ Y) )   → (A Aᵀ)^q A Ω
        row(A):  Y = Aᵀ Ω;  repeat q× :  Y = orth( Aᵀ (A Y) )   → (Aᵀ A)^q Aᵀ Ω
        row(B):  Y = Bᵀ Ω;  repeat q× :  Y = orth( Bᵀ (B Y) )   → (Bᵀ B)^q Bᵀ Ω

    Each step multiplies the captured spectrum by another factor of the singular
    values, so the subspace aligns with the dominant directions far better —
    strictly lower Frobenius error at the same ``M``. ``q=0`` reduces exactly to
    ``rsvd``. The re-orthonormalization between steps keeps the iteration numerically
    stable (round-off would otherwise collapse ``Y`` onto the top singular vector).

    The extra cost is ``q`` more streamed passes over A/B (``~4 q N² M`` FLOPs,
    reported honestly in :meth:`basis_flops`) — still ``O(N² M) ≪ O(N³)``, so the
    strategy can still dominate exact while gaining accuracy. The sketches stream
    (``stream_gemm_*``), so A/B may be disk-backed memmaps and the basis stage
    honours the same ``--vram-fraction`` budget as compress/reconstruct.
    """

    name = "power-rsvd"

    def __init__(self, seed: int = 0, q: int = 2):
        super().__init__(seed=seed)
        if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 0:
            raise ValueError(f"q (power-iteration steps) must be a non-negative integer, got {q!r}")
        self.q = int(q)

    def basis(self, n, m, backend, dtype, A=None, B=None, frac=None):
        if A is None or B is None:
            raise ValueError("power-rsvd transform needs A and B")
        from .subspace import (
            _DEFAULT_ROW_BLOCK_FRACTION,
            stream_gemm_left_t,
            stream_gemm_right,
        )

        # Honour the strategy's VRAM budget for the sketch row-blocks, exactly as
        # rsvd/compress/reconstruct do -- otherwise the basis stage silently uses
        # the 0.3 default and can OOM at a low --vram-fraction.
        if frac is None:
            frac = _DEFAULT_ROW_BLOCK_FRACTION

        xp = backend.xp
        base, rem = divmod(m, 3)
        widths = [base + (1 if i < rem else 0) for i in range(3)]
        rng = np.random.default_rng(self.seed)

        def omega(w):
            return backend.to_device(
                rng.standard_normal((n, w)).astype(dtype, copy=False)
            )

        def power(Y, X, *, transpose_first):
            # Sharpen Y toward the dominant singular subspace of X (transpose_first
            # False -> range of X; True -> range of Xᵀ) with q re-orthonormalized
            # subspace-iteration steps. Both products stream, so X may be a memmap.
            for _ in range(self.q):
                if transpose_first:                                   # Y <- Xᵀ (X Y)
                    Z = stream_gemm_right(X, Y, backend, dtype, frac)     # X Y
                    Y = stream_gemm_left_t(X, Z, backend, dtype, frac)    # Xᵀ Z
                else:                                                 # Y <- X (Xᵀ Y)
                    Z = stream_gemm_left_t(X, Y, backend, dtype, frac)    # Xᵀ Y
                    Y = stream_gemm_right(X, Z, backend, dtype, frac)     # X Z
                Y = self._orthonormalize(Y, backend)                  # stabilize
            return Y

        parts = []
        if widths[0]:
            Y = stream_gemm_right(A, omega(widths[0]), backend, dtype, frac)   # col(A): A Ω
            parts.append(power(Y, A, transpose_first=False))
        if widths[1]:
            Y = stream_gemm_left_t(A, omega(widths[1]), backend, dtype, frac)  # row(A): Aᵀ Ω
            parts.append(power(Y, A, transpose_first=True))
        if widths[2]:
            Y = stream_gemm_left_t(B, omega(widths[2]), backend, dtype, frac)  # row(B): Bᵀ Ω
            parts.append(power(Y, B, transpose_first=True))

        Y = xp.concatenate(parts, axis=1)      # (n, m)
        return self._orthonormalize(Y, backend)  # (n, m) orthonormal columns

    def basis_flops(self, n, m):
        # rsvd's sketch + final QR (2 n² m + 2 n m²), PLUS q subspace-iteration
        # steps. Each step is two n×n · n×w products (Xᵀ Y then X Z) over widths
        # summing to m -> 4 n² m, plus a re-orthonormalizing thin QR bounded by
        # 2 n m². Recomputed every call (depends on A, B), so not amortizable.
        rsvd_cost = 2.0 * n * n * m + 2.0 * n * m * m
        return rsvd_cost + self.q * (4.0 * n * n * m + 2.0 * n * m * m)


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


for _cls in (RandomizedSVDTransform, NystromTransform, PowerIterationTransform):
    register_transform(_cls.name, _cls)
