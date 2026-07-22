# Attention Playground

This folder is a **local prototype area** for the repo's next evaluation
direction: attention-shaped workloads.

It is deliberately separate from the shipped `matmul/`, `strategy/`, and
`eval/` pipeline. The goal is to let you test one honest idea on a real GPU
before deciding how to make attention a first-class scored track.

## What is here

- `reference.py`
  Exact scaled dot-product attention:
  `softmax(QKᵀ / sqrt(d)) V`
- `hybrid.py`
  A first hybrid operator:
  exact local-window attention + cheap FFT-based global mixing
- `spec.py`
  Shared `AttentionSpec` describing one benchmark shape
- `data.py`
  Synthetic Q/K/V generation for that spec
- `benchmark.py`
  Compare exact vs hybrid on one device and report quality / latency / VRAM

## Why this is separate

The current main repo measures:

- exact matmul
- approximate subspace matmul

Attention is related, but it is not the same operator:

- it uses tensor-shaped Q/K/V inputs
- it contains a nonlinear softmax step
- it often needs masking / causality / local windows

So this folder is a safe staging area, not a claim that the final repo
architecture should permanently look like this.

## Local commands

CPU-safe tests:

```bash
uv sync --extra test
uv run --extra test python -m pytest tests/test_attention_playground.py -q
```

Small benchmark on a GPU machine:

```bash
uv sync --extra test --extra gpu
uv run --extra gpu python -m attention.benchmark \
  --seq 1024 --heads 4 --dim 32 --dtype fp16 --window 128
```

JSON output:

```bash
uv run --extra gpu python -m attention.benchmark \
  --seq 1024 --heads 4 --dim 32 --dtype fp16 --window 128 --json
```

## What "success" means for this stage

At this stage, success is only:

1. the operator runs correctly,
2. the benchmark compares against an exact baseline,
3. the RTX 5070 Ti can measure quality, latency, and VRAM honestly.

This stage does **not** need to beat exact attention yet. It exists to create a
clean baseline for future hybrid and track-integration work.
