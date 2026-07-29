# Cuda-Compute-OSS (CCO)

CCO is an evidence-first project for efficient long-context attention on
consumer NVIDIA GPUs.

The long-term research target is a hybrid operator with exact local attention,
a learned causal FFT global path, and a selective retrieval path only when
recall evidence justifies it. CCO does **not** currently claim that FFT replaces
attention or improves an LLM.

## Current phase

| Field | Value |
|---|---|
| Phase | Attention foundation |
| Benchmark ID | `attention-foundation-v1.1-rtx5070ti` |
| Status | **DRAFT — miner PRs closed** |
| Reference GPU | NVIDIA GeForce RTX 5070 Ti |
| Goal | Improve exact attention latency or memory without correctness loss |
| Merge automation | Locked pending hardened-protocol calibration and certification |

The protected manifest is
[`benchmarks/attention-foundation-v1.1-rtx5070ti.json`](benchmarks/attention-foundation-v1.1-rtx5070ti.json).
The previous calibrated v1 manifest remains historical evidence. Version 1.1
changes the measurement protocol and must be recalibrated before mining reopens.

## What exists

- `matmul/`: exact in-core and tiled GEMM with bounded-memory paths.
- `strategy/`: retained legacy research on synthetic square matrices.
- `attention/`: exact, local, spectral, correlation, and landmark prototypes.
- `eval/`: the legacy square evaluator plus the protected attention
  evaluator and PR policy.

Legacy square-GEMM results remain historical evidence. They are not evidence
of attention quality, model quality, prefill speed, or decode speed.

## First benchmark

The attention-foundation phase keeps output exact and asks whether a candidate
can reduce attention latency or memory. Correctness is a hard gate, not a
weighted score.

The draft benchmark contains nine GPU workload configurations:

- four causal prefill workloads at sequence lengths 1K, 2K, 4K, and 8K;
- three one-token decode workloads with 1K, 4K, and 8K KV contexts;
- one ragged causal guard and one non-causal guard.

Each official comparison uses the same RTX 5070 Ti environment, 10 warm-ups,
30 measured repetitions, raw CUDA-event timings, peak incremental VRAM, and
balanced `main → PR → PR → main` timing shards. Independent OS-random input
seeds are generated immediately before each call and published afterward.
Every warm-up and repetition uses fresh inputs, every timed output is checked,
and contributor code is imported only after protected reference measurement.
Prefill and decode are decided separately.

The references are:

- explicit fp32 attention for affordable mathematical-oracle cases;
- PyTorch scaled-dot-product attention as the production reference;
- the current `main` implementation as the performance frontier.

## Contribution policy

Mining is currently closed while v1.1 is calibrated and adversarially
certified. Feature proposals may be opened, but maintainers will not apply
`status:phase-approved` until activation. The planned implementation scope is:

- rectangular and batched attention-shaped multiplication;
- `QK^T` and `PV` paths used by attention;
- exact prefill and one-token decode improvements;
- layout, workspace, and peak-memory improvements;
- exact blockwise implementations;
- focused tests for the implementation change.

Bug-fix, documentation, cleanup, refactor-only, legacy transform, evaluator,
benchmark, workflow, and policy PRs are not miner contribution lanes. Changes
to protected infrastructure are made separately by maintainers.

After activation, a current-phase PR must:

1. use the feature template;
2. declare the exact active benchmark ID;
3. close an open issue carrying `status:phase-approved`;
4. touch only current-phase implementation/test paths;
5. pass normal CI and protected GPU correctness;
6. beat current `main` beyond calibrated noise without a protected regression.

Contributor GPU numbers are diagnostic. Only the protected same-machine
evaluation can label, close, or merge a PR.

## Automatic decision sequence

```text
admission and protected-path checks
              |
              v
CPU CI and correctness tests
              |
              v
RTX 5070 Ti: main -> PR -> PR -> main
              |
              v
correctness, latency, VRAM, and significance decision
       |              |                 |
     reject         no gain           admit
       |              |                 |
     close          close           merge one
                                        |
                                        v
                              test merged main
                                        |
                              re-evaluate next PR
```

Automatic processing requires an explicit benchmark-ID confirmation, verifies
the queued head SHA and current `main`, and halts if
post-merge validation fails. After admitting one PR, later candidates are
locally merged with the new `main` for evaluation; conflicts are blocked, and
neither the contributor branch nor its recorded head SHA is rewritten.

Terminal decisions are written to an append-only attention ledger with raw
artifact hashes and projected separately to `dashboard/attention-results.json`
on the bot state branch. Attention results never enter the legacy square-GEMM
leaderboard.

## Development setup

```bash
git clone https://github.com/zeokin/Cuda-Compute-OSS.git
cd Cuda-Compute-OSS
uv sync --extra test
uv run python -m compileall -q matmul strategy eval attention tests examples
uv run --extra test python -m pytest tests/ strategy/tests/ eval/tests/ -q
uv run python -m strategy.smoke
```

Small attention-prototype diagnostic:

```bash
uv sync --extra test --extra gpu
uv run --extra gpu python -m attention.benchmark \
  --seq 1024 --heads 4 --dim 64 --dtype fp16 --window 128 --json
```

This prototype command is not an official scorecard.

Protected benchmark in one clean RTX 5070 Ti checkout:

```bash
python -m eval.attention_benchmark --official --json
```

Manual queue preview on the Windows CUDA environment:

```bat
python -m eval.attention_batch --limit 0 --active-python
```

The first command previews the queue. Processing is intentionally unavailable
while v1.1 is draft; `--process` refuses to run.

## Repository trust boundary

Contributor PRs cannot modify `eval/`, `benchmarks/`, `.github/`, `dashboard/`,
public policy, or package configuration. The evaluator records the manifest
hash, commit, post-run seed, environment, raw samples, timed-output checks, and
workload results. A stale result, dirty checkout, repeated-input cache, or
un-restored runtime mutation cannot authorize a merge.

Detailed roadmap, calibration procedure, held-out cases, Windows operations,
and recovery instructions are maintainer material and are not published as a
miner roadmap. This README is the public project and phase source of truth.

## License

MIT. See [LICENSE](LICENSE).
