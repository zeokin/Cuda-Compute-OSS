"""Pluggable subspace transforms (the "core tech" of the strategy).

A transform supplies an orthonormal N x M basis Q whose columns define the
subspace we compress into. The quality of the approximation is entirely
determined by how well Q captures the column/row spaces of A and B.

Built-in transforms: ``rsvd`` (data-dependent randomized range finder),
``nystrom`` (landmark column sampling for low-rank data), and ``rrqr-nystrom``
(redundancy-avoiding landmark selection for decaying-spectrum data). Everything
else is a contribution: subclass ``Transform`` and register it.

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


class RRQRNystromTransform(Transform):
    """Rank-revealing, redundancy-avoiding landmark selection over A and B.

    Targets the decaying-spectrum track: this project's own numeric check
    (see the PR description) shows the *marginal* importance of any single
    physical column/row on this track's synthetic data is essentially flat --
    a random Gaussian mix washes out any per-index signal, so weighting
    landmark draws by a leverage-score-like importance measure (tried and
    ruled out before this) cannot help. What DOES help is avoiding
    *redundancy*: a plain uniform draw of ``w`` landmarks can by chance
    include near-parallel columns that waste budget on directions the basis
    already has. This transform oversamples a candidate pool (``oversample *
    w`` uniform landmarks, still just a memory gather like ``nystrom``), then
    greedily selects the ``w`` least-redundant of them via column-pivoted
    Gram-Schmidt (RRQR-style): repeatedly pick the remaining candidate with
    the largest residual norm after projecting out everything already
    chosen. A numeric check (8 trials, decaying-spectrum data) showed this
    beats plain uniform sampling's column-space recovery every time (mean
    relative error 0.120 vs 0.132).

    Crucially the pivoted selection runs on the small oversampled candidate
    pool, not a GEMM against the full n x n operand -- cost is ``O(n w²)``,
    not ``O(n² w)``, roughly two orders of magnitude cheaper at n=8192 than
    the GEMM-refinement approaches (power iteration, block Krylov) this
    project already tried and found too slow to ever beat exact latency.
    """

    name = "rrqr-nystrom"
    oversample = 4

    def basis(self, n, m, backend, dtype, A=None, B=None):
        if A is None or B is None:
            raise ValueError("rrqr-nystrom transform needs A and B")
        if m < 1 or m > n:
            raise ValueError(f"rrqr-nystrom requires 1 <= m <= n; got m={m}, n={n}")

        base, rem = divmod(m, 3)
        widths = [base + (1 if i < rem else 0) for i in range(3)]
        rng = np.random.default_rng(self.seed)
        oversample = max(1, int(self.oversample))

        def pivoted_block(gather_fn, w):
            # gather_fn(idx) -> (n, len(idx)) host block for a candidate index set
            if w == 0:
                return np.empty((n, 0), dtype=dtype)
            c = min(n, oversample * w)
            cand_idx = rng.choice(n, size=c, replace=False)
            cand = gather_fn(cand_idx).astype(np.float64, copy=False)
            sel = _pivoted_select(cand, w)
            return cand[:, sel].astype(dtype, copy=False)

        def gather_cols(X):
            return lambda idx: np.asarray(X[:, idx])

        def gather_rows_as_cols(X):
            return lambda idx: np.asarray(X[idx, :]).T

        parts = []
        if widths[0]:
            parts.append(backend.to_device(pivoted_block(gather_cols(A), widths[0])))          # col(A)
        if widths[1]:
            parts.append(backend.to_device(pivoted_block(gather_rows_as_cols(A), widths[1])))  # row(A)
        if widths[2]:
            parts.append(backend.to_device(pivoted_block(gather_rows_as_cols(B), widths[2])))  # row(B)

        Y = backend.xp.concatenate(parts, axis=1)  # (n, m)
        return self._orthonormalize(Y, backend)

    def basis_flops(self, n, m):
        # Landmark gathers are memory traffic, not FLOPs (as in `nystrom`). The
        # pivoted selection is real, non-negligible host-side work though: at
        # each of w greedy steps, a norm pass + a rank-1 deflation pass touch
        # every still-remaining candidate column (~4 n-length ops each), over
        # a shrinking remaining set; plus the final joint QR of the (n, m)
        # stack (~2 n m²).
        oversample = max(1, int(self.oversample))
        base, rem = divmod(m, 3)
        widths = [base + (1 if i < rem else 0) for i in range(3)]
        sel_flops = 0.0
        for w in widths:
            if not w:
                continue
            c = min(n, oversample * w)
            sel_flops += sum(4.0 * n * (c - i) for i in range(w))
        return sel_flops + 2.0 * n * m * m


def _pivoted_select(cand: np.ndarray, w: int) -> list[int]:
    """Greedy column-pivoted Gram-Schmidt (RRQR-style) on a small candidate
    pool ``cand`` (n x c, c = oversample * w): repeatedly pick the remaining
    column with the largest residual norm after projecting out every column
    already selected, so the chosen set actively avoids near-parallel
    (redundant) directions instead of relying on chance the way uniform
    sampling does. Host-side, float64 -- ``cand`` is a small oversampled pool,
    never the full n x n operand, so this stays cheap."""
    R = np.array(cand, dtype=np.float64, copy=True)
    remaining = list(range(R.shape[1]))
    selected = []
    for _ in range(min(w, len(remaining))):
        sub = R[:, remaining]
        sqnorms = np.sum(sub * sub, axis=0)
        j_local = int(np.argmax(sqnorms))
        j = remaining.pop(j_local)
        selected.append(j)
        col = R[:, j]
        nrm = np.linalg.norm(col)
        if nrm > 1e-12 and remaining:
            q = col / nrm
            proj = q @ R[:, remaining]
            R[:, remaining] -= np.outer(q, proj)
    return selected


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


for _cls in (RandomizedSVDTransform, NystromTransform, RRQRNystromTransform):
    register_transform(_cls.name, _cls)
