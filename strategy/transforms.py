"""Pluggable subspace transforms (the "core tech" of the strategy).

A transform supplies an orthonormal N x M basis Q whose columns define the
subspace we compress into. The quality of the approximation is entirely
determined by how well Q captures the column/row spaces of A and B.

Built-in transforms: ``rsvd`` (data-dependent randomized range finder) and
``nystrom`` (landmark column sampling for low-rank data). Everything else is a
contribution: subclass ``Transform`` and register it.

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

        # Preallocate the (n, m) assembly buffer and stream each sketch into a
        # column slice, releasing the temporary immediately. The previous
        # approach kept all parts alive then ``concatenate``'d them into another
        # (n, m) buffer -- a ~2x (n, m) spike that was never charged against
        # ``frac`` and could OOM on MPS right after the sketches succeeded (#307).
        # Charging Y as extra_fixed_bytes on every sketch keeps the stream
        # budgets honest for the buffer that will hold the final range.
        item = np.dtype(dtype).itemsize
        Y = xp.empty((n, m), dtype=dtype)
        y_bytes = n * m * item
        col = 0
        if widths[0]:
            w = widths[0]
            part = stream_gemm_right(
                A, omega(w), backend, dtype, frac,
                extra_fixed_bytes=y_bytes,
            )   # col(A): A Ω
            Y[:, col:col + w] = part
            del part
            col += w
        if widths[1]:
            w = widths[1]
            part = stream_gemm_left_t(
                A, omega(w), backend, dtype, frac,
                extra_fixed_bytes=y_bytes,
            )   # row(A): Aᵀ Ω
            Y[:, col:col + w] = part
            del part
            col += w
        if widths[2]:
            w = widths[2]
            part = stream_gemm_left_t(
                B, omega(w), backend, dtype, frac,
                extra_fixed_bytes=y_bytes,
            )   # row(B): Bᵀ Ω
            Y[:, col:col + w] = part
            del part
            col += w

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


class SparseSignSketchTransform(Transform):
    """Sparse-sign (OSNAP-style) sketch over A and B.

    Middle ground between ``nystrom`` (one landmark column per direction —
    cheap, but a lone column can miss structure) and ``rsvd`` (dense Gaussian
    sketch — accurate, but ``O(2 N² M)`` GEMMs). Each output direction is a
    **signed sum of ``s`` randomly chosen columns** (resp. rows), i.e.
    ``A @ Ω`` with a sparse sign matrix ``Ω`` (``s`` nonzeros per column).
    Mixing recovers more of the range per direction than a single landmark,
    while the host-side gather stays in ``nystrom``'s cost class (no dense
    sketch GEMM, no full-matrix device staging).

    Spaces sketched: col(A), row(A), row(B) — the three necessary for
    ``Ĉ = P A P B P`` (col(B) is redundant; same split as ``rsvd``). Then a
    thin QR. Analytic cost ``~2 N M² + (s-1) N M`` sits between ``nystrom``
    and ``rsvd``.

    ``s`` defaults to 2: enough mixing to beat plain landmarks on decaying
    spectra in practice, without the latency tax of larger sparsity (or of
    copying A/B onto the device just to ``index_select``).
    """

    name = "sparsesign"
    sparsity = 2

    def basis(self, n, m, backend, dtype, A=None, B=None, frac=None):
        if A is None or B is None:
            raise ValueError("sparsesign transform needs A and B")
        if m < 1 or m > n:
            raise ValueError(f"sparsesign requires 1 <= m <= n; got m={m}, n={n}")

        s = int(self.sparsity)
        if s < 1:
            raise ValueError(f"sparsesign sparsity must be >= 1; got {s}")

        base, rem = divmod(m, 3)
        widths = [base + (1 if i < rem else 0) for i in range(3)]
        rng = np.random.default_rng(self.seed)
        xp = backend.xp
        # Preallocate the assembly buffer and stream each part into a column
        # slice (same pattern as rsvd). Avoids a ~2x (n, m) concatenate spike
        # after the gathers succeed.
        Y = xp.empty((n, m), dtype=dtype)
        col = 0
        scale = 1.0 / np.sqrt(float(s))

        def mix_columns(X, w):
            # Host gather + signed sum — O(n * w * s) memory traffic, no GEMM.
            # Sampling with replacement keeps index generation O(w*s) in n.
            # Stay in ``dtype`` (same as nystrom) — avoid a float64 round-trip
            # that only adds host work before the device upload.
            idx = rng.integers(0, X.shape[1], size=(w, s))
            signs = rng.choice(np.array([-1.0, 1.0], dtype=dtype), size=(w, s))
            gathered = np.asarray(X[:, idx]).astype(dtype, copy=False)  # (n, w, s)
            return (gathered * signs).sum(axis=2) * np.asarray(scale, dtype=dtype)

        def mix_rows_as_cols(X, w):
            # Mix rows of X as columns of Xᵀ without materializing a full transpose.
            idx = rng.integers(0, X.shape[0], size=(w, s))
            signs = rng.choice(np.array([-1.0, 1.0], dtype=dtype), size=(w, s))
            # X[idx, :] -> (w, s, n); move axis to (n, w, s)
            gathered = np.asarray(X[idx, :]).astype(dtype, copy=False)
            if gathered.ndim == 2:
                # w*s flat index path shouldn't happen with (w,s) idx, but be safe
                gathered = gathered.T.reshape(X.shape[1], w, s)
            else:
                gathered = np.moveaxis(gathered, -1, 0)  # (n, w, s)
            return (gathered * signs).sum(axis=2) * np.asarray(scale, dtype=dtype)

        if widths[0]:
            w = widths[0]
            part = backend.to_device(mix_columns(A, w))
            Y[:, col:col + w] = part
            del part
            col += w
        if widths[1]:
            w = widths[1]
            part = backend.to_device(mix_rows_as_cols(A, w))
            Y[:, col:col + w] = part
            del part
            col += w
        if widths[2]:
            w = widths[2]
            part = backend.to_device(mix_rows_as_cols(B, w))
            Y[:, col:col + w] = part
            del part
            col += w

        return self._orthonormalize(Y, backend)

    def basis_flops(self, n, m):
        # Thin QR (~2 n m²) plus (s-1) real additions per output element for the
        # signed blend (nystrom's single-column copy has no add). Gather itself
        # is memory traffic, not FLOPs.
        s = int(self.sparsity)
        return 2.0 * n * m * m + float(max(s - 1, 0)) * n * m


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


for _cls in (RandomizedSVDTransform, NystromTransform, SparseSignSketchTransform):
    register_transform(_cls.name, _cls)
