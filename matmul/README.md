# `matmul` — normal (exact) engine

Compute **C = A × B** for square `n × n` matrices on the GPU, where `n` is
configurable and may be **very large** (e.g. 128k, 256k) — far beyond what fits
in a single GPU's memory. Written in **Python** (PyTorch → GPU), with automatic
**out-of-core tiling**. Compute is **GPU-only** (CUDA or Apple MPS).

This is the **normal (exact)** baseline of [CCO](../README.md) — the frontier a
smart strategy must beat on cost without losing accuracy.

## Two methods, two folders

CCO provides two **independent, self-contained** ways to multiply. Each lives in
its own folder and neither imports the other.

| folder | method | what it does | result |
|---|---|---|---|
| [`matmul/`](.) | **normal (exact)** | full tiled/streamed `O(N³)` GPU multiply | exact |
| [`strategy/`](../strategy/) | **smart (subspace)** | compress `(N,N)→(M,M)` → multiply → reconstruct, `O(N²M)` | approximate (see [strategy/README.md](../strategy/README.md)) |

The rest of this file documents the **normal (exact)** method in `matmul/`. For
the smart method, see [strategy/README.md](../strategy/README.md).

```bash
python -m matmul   --n 8192 --verify                      # exact
python -m strategy --n 8192 --transform rsvd --verify  # smart
```

## The core problem: it doesn't fit

A square matrix in FP32 costs `n² × 4` bytes. You need three of them (A, B, C):

| n | one matrix (FP32) | A + B + C | compute (2n³) |
|---|---|---|---|
| 1k | 4 MB | 12 MB | 2.0 GFLOP |
| 16k | 1 GB | 3 GB | 8.2 TFLOP |
| 32k | 4 GB | 12 GB | 66 TFLOP |
| 128k | **65 GB** | **196 GB** | 4.2 PFLOP |

An A100 has **80 GB**. Small sizes (the default `n = 8192` is ~0.8 GB
for A+B+C) run **in-core** in a single GPU GEMM. Larger ones don't fit — already
at `n = 128k`, A+B+C is 196 GB — so this system streams them. It:

1. Stores A, B, C as **memory-mapped files on disk** (never fully in RAM).
2. Streams **T × T tiles** to the GPU and computes
   `C[i,j] = Σ_k A[i,k] @ B[k,j]` with `torch.bmm`.
3. Falls back to a single GPU GEMM ("in-core") automatically when the whole
   thing *does* fit.

Tips to reach the biggest `n`: use `--dtype fp16` (half the memory, tensor-core
speed) and make sure you have enough **disk** (196 GB+ for 128k fp32; use fp16
to roughly halve it).

## Install

```bash
pip install -r requirements.txt          # numpy + torch
```

Compute is **GPU-only** — a CUDA (`pip install torch`) or Apple-MPS device is
required. With no GPU, the engine raises a clear error.

## Backend

The engine talks only to [`backend.py`](backend.py): a single **PyTorch GPU**
backend that auto-selects **CUDA → Apple MPS**. `backend.xp` is a tiny
NumPy-compatible shim so the tiling code (`xp.zeros / @ / .astype /
xp.linalg.qr`) runs on torch tensors unchanged. Every tile product in this
**normal (exact) engine** goes through **`torch.bmm`** (a 2-D tile is run as a
batch of one) via `Backend.matmul`; the smart engine uses `torch.matmul`.

## Use it — CLI

```bash
# Default size, checked against a float64 reference (n defaults to 8192):
python -m matmul --n 8192 --dtype fp32 --verify

# Benchmark a size that fits on the GPU:
python -m matmul --n 20000 --dtype fp16

# Huge, out-of-core, disk-backed:
python -m matmul --n 128000 --dtype fp16 --storage disk \
                 --workdir /data/mm --keep

# Exercise the tiled/streaming path on a small size (no giant matrix needed):
python -m matmul --n 1024 --force-tiled --tile 256 --verify
```

Key flags: `--n` (size), `--dtype {fp16,fp32,fp64}`, `--device` (GPU index),
`--tile T` (tile edge; default auto from free VRAM), `--vram-fraction`,
`--storage {auto,ram,disk}`, `--workdir`, `--verify`.

## Use it — Python API

```python
import numpy as np
from matmul import matmul

A = np.random.rand(1024, 1024).astype("float32")
B = np.random.rand(1024, 1024).astype("float32")
C = matmul(A, B)                      # runs on the GPU (CUDA/MPS)

# Large out-of-core run with full control:
from matmul import Config, runner
cfg = Config(dtype="fp16", storage="disk", workdir="/data/mm",
             vram_fraction=0.6)
info = runner.run(128000, cfg, fill="random", keep=True)
print(info["gflops"], "GFLOP/s")
```

> The **smart (subspace)** method now lives in its own standalone folder —
> see [strategy/README.md](../strategy/README.md).

## How it works

```
matmul/
  config.py     Config dataclass — dtype, tile, device, storage, ...
  backend.py    PyTorch GPU backend (CUDA/MPS); NumPy-compat shim, transfers
  storage.py    memmap vs RAM allocation; block-wise random/iota fill (host)
  gemm.py       exact engine: in-core / tiled-sync + auto tiling (torch.bmm)
  runner.py     generate A,B, run, time, verify, report GFLOP/s
  cli.py        python -m matmul
```

**Tiling.** `C[i,j] = Σ_k A[i,k] @ B[k,j]` over `T × T` blocks. `T` is auto-sized
so `acc + (A,B tiles)` fits in `vram_fraction` of free VRAM. Ragged edges (n not
divisible by T) are handled. Each A row-panel is cached in host RAM for the
duration of its column sweep to cut disk re-reads.

**Precision.** `fp16` inputs accumulate across tiles in `fp32` for accuracy;
`fp32`/`fp64` accumulate in kind.

## Test

```bash
python tests/test_correctness.py         # or: python -m pytest tests/ -q
```

Tests validate the blocking math (ragged tiles, fp16 accumulation) on the GPU.
They **skip** when no CUDA/MPS device is present.

## Status / notes

- Compute is **GPU-only** (PyTorch on CUDA or Apple MPS); with no GPU the engine
  raises a clear error and the tests skip.
- Validate throughput and accuracy on your target card.
- `bf16` is not exposed (NumPy has no native bf16); use `fp16`.
