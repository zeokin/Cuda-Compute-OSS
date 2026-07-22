# Attention Track Plan

This file defines how the local `attention/` prototype should evolve into a
first-class repo track without weakening the current evaluation discipline.

## Goal

Add an `attention-shaped` evaluation track that keeps the same testing
philosophy as the existing matmul track:

- fast local smoke test
- normal PR CI
- maintainer-controlled RTX 5070 Ti validation
- exact baseline on the same inputs
- latency / VRAM / quality measured together

## What changes

The workflow shape does not change. The operator under test changes.

Today:

- exact baseline: `A @ B`
- candidate: subspace matmul
- quality metric: bounded Frobenius accuracy

Future attention track:

- exact baseline: exact scaled dot-product attention
- candidate: hybrid local-exact + global structured operator
- quality metric: MSE and/or relative output error against exact attention

## Track definition

### Track name

`attention-shaped`

### Exact baseline

Exact scaled dot-product attention on identical Q/K/V:

`softmax(QKᵀ / sqrt(d)) V`

### Candidate family

At first:

- local exact window attention
- FFT-based global mixer
- weighted hybrid of the two

Later:

- learned spectral filters
- per-head hybrid weights
- model-derived local/global operator variants

### Inputs

First stage:

- synthetic Q/K/V tensors
- fixed benchmark shapes
- pinned dtype
- fresh seed per validator run

Later stage:

- Gemma-derived shapes / model-derived tensors

### Quality metric

First stage:

- MSE vs exact attention output
- relative Frobenius error
- simple `accuracy_proxy = max(0, 1 - rel_err)` for dashboards

### Cost metrics

Keep the same style as the current repo:

- latency
- peak VRAM
- scaling as sequence length grows

## Repo structure proposal

### Stage A: prototype

Current local-only state:

- `attention/`
- `tests/test_attention_playground.py`

### Stage B: real track bootstrap

Add:

- `attention/tests/`
- `eval/attention_evaluator.py` or a track-aware evaluator abstraction
- validator command for exact-vs-hybrid attention
- dashboard support for track-specific results

### Stage C: first-class track

Expected future pieces:

- track registry
- one PR targets one track
- per-track exact baseline
- per-track quality metric
- per-track frontier

## Local / CI / validator split

### Local

- shape sanity
- no-NaN checks
- local branch equals exact when window covers full sequence
- optional small GPU benchmark

### PR CI

- tiny CPU-safe tests only
- no large GPU dependency required for default CI

### RTX 5070 Ti validator

- exact attention baseline
- candidate hybrid operator
- fixed benchmark shapes
- fresh seed per run
- publish latency / VRAM / quality together

## What not to do

- Do not merge attention scoring into the current matmul score.
- Do not let an attention win cancel a matmul loss.
- Do not let miners define new scoring logic directly inside protected paths
  without first accepting the track design.

## Recommended next implementation steps

1. Keep `attention/` as a prototype area while you test the operator locally.
2. Decide the first pinned benchmark shapes for RTX 5070 Ti runs.
3. Choose the first official attention quality metric.
4. Add a track-aware result model in the dashboard / ledger path.
5. Only then wire attention into the maintainer GPU validator.
