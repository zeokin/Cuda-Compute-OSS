# Smart (subspace) matrix multiply

An **approximate** matrix multiply that trades accuracy for speed by working in
a small subspace. Standalone package — it does **not** import the sibling
[`matmul/`](../matmul/) (exact) package; it ships its own backend + storage.

## The pipeline

For square `n × n` A, B and an orthonormal `n × M` basis `Q` (M ≪ n):

```
(N,N) --compress-->  Ã = Qᵀ A Q,  B̃ = Qᵀ B Q     (M,M)
      --compute--->  C̃ = Ã @ B̃                    (M,M)   <- the cheap core
      --reconstruct-> C = Q C̃ Qᵀ                   (N,N)
```

Cost drops from `O(N³)` to `O(N²M)` (FLOP ratio ≈ `3M/N`). The projections
stream over A/B one row-block at a time, so this stays out-of-core (memmap).

## Honesty about accuracy — READ THIS

The result equals `P·A·P·B·P` with projector `P = QQᵀ`. It is **exact only when
M = N**, or when A and B genuinely live in the subspace `Q` captures (low rank /
smooth structure). On **full-rank random data with M ≪ N the error is ~100%** —
that is expected, not a bug. So this method is for **compressible** data. The
runner always reports the reconstruction error next to the timing; check it.

When it wins: **huge N** (where exact `O(N³)` is infeasible), **M ≪ N**, and
data that is actually low-rank/smooth. When it loses: small N, or full-rank data
(and on CPU at small N the exact BLAS call is simply faster).

## The transform is the pluggable "core tech"

`Q` comes from a **transform**, chosen with `--transform` (or `Config(transform=...)`)
and swappable/updatable via a registry:

| transform | kind | best for |
|---|---|---|
| `rsvd`   | **data-dependent** range finder over A and B | general low-rank data (accurate) |

`rsvd` is the **only** built-in transform. New transforms are the contribution
surface — subclass `Transform` and register your own below.

Register your own (the updatable hook):

```python
from strategy import Transform, register_transform, subspace_matmul, Config

class MyTransform(Transform):
    name = "mine"
    def basis(self, n, m, backend, dtype, A=None, B=None):
        Q = ...                     # (n, m), ORTHONORMAL columns, on backend.xp
        return Q

register_transform("mine", MyTransform)
C = subspace_matmul(A, B, config=Config(transform="mine", rank_m=256))
```

## Use it

CLI:

```bash
# full-rank 12000 (the honest hard case): reconstruction error is large by design.
python -m strategy --n 12000 --transform rsvd --fill random --verify

# smart multiply on compressible data (where it works), report error:
python -m strategy --n 12000 --transform rsvd --fill lowrank --data-rank 16 --verify

# normal (exact) vs smart, side by side on the same inputs:
python -m strategy --n 12000 --compare --transform rsvd --fill lowrank --data-rank 16
```

(`M` defaults to `n // 8` = 1500 at `n = 12000`; set it with `--rank-m`.)

Key flags: `--n`, `--dtype {fp16,fp32,fp64}`, `--rank-m M`, `--transform`,
`--compare`, `--fill {lowrank,random,iota,zeros}`, `--data-rank`, `--storage`,
`--device`, `--verify`.

Compute is **GPU-only** — PyTorch on **CUDA → Apple MPS**. Every product in this
**smart engine** goes through **`torch.matmul`** (the normal `matmul/` engine
uses `torch.bmm`). With no GPU the CLI prints a clear error.

Python API:

```python
import numpy as np
from strategy import subspace_matmul, Config

A = ...  # (n, n), ideally low-rank / smooth
B = ...
C = subspace_matmul(A, B, config=Config(transform="rsvd", rank_m=256))
```

## Layout

```
strategy/
  config.py     Config — dtype, rank_m (M), transform, device, storage, ...
  backend.py    PyTorch GPU backend (CUDA/MPS)   (self-contained copy)
  storage.py    memmap/RAM alloc; random/lowrank/iota fill (self-contained copy)
  transforms.py pluggable bases Q (rsvd built-in) + registry
  subspace.py   streaming compress/reconstruct, subspace multiply, exact baseline
  runner.py     generate A,B, run, verify, compare
  cli.py        python -m strategy
  tests/        correctness + exactness/accuracy tests (GPU; skip if none)
  examples/     run_example.py
```

## Test

```bash
python strategy/tests/test_subspace.py      # or: python -m pytest strategy/tests -q
```

The tests cover the streaming primitives, the exact baseline, exactness at
`M = N`, `rsvd` recovery of low-rank products, a custom-basis transform, and the
transform registry. They run on the GPU (PyTorch) and **skip** when no CUDA/MPS
device is present.
