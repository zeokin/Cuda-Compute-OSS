# Benchmarks

What the harness measures, how a submission is scored, and the current seed-champion baselines.
All numbers come from real GPUs running the unmodified, locked harness (`benchmark.py`) — no
simulators, no estimates.

> **The scored axis is speedup vs the current champion kernel, not vs PyTorch.** The PyTorch
> reference in `references/` is the *correctness oracle only*. A challenger wins a track when it is
> correct (5-stage hard gate), uses no more VRAM, and is faster than the standing champion with
> statistical significance (one-sided Mann-Whitney U + a margin). See [DESIGN.md](DESIGN.md) §3.

---

## How a number is produced

```bash
uv run benchmark.py            # 5-stage correctness + roofline on the published self-score seed (42)
uv run benchmark.py --score    # the competition latency SAMPLE (n_blocks block-means) on the primary size
uv run benchmark.py --blob     # the bound score blob (sample + correctness + identity hashes)
```

- **Latency sample:** `n_blocks = 30` block-mean latencies on the primary size + dtype, with
  rotating input buffers (kills warm-L2 / memoize-by-pointer), fused correctness on two distinct
  buffers, and an output-vs-input alias guard (`run_isolated`; the in-process `run_scored_sample`
  fallback implements the same check but is not authoritative).
- **Win decision** (`cco/significance.py`): the challenger beats the champion only if it is
  *significantly* faster (`p < p_value_threshold`) **and** faster by ≥ `min_improvement_pct` — both
  thresholds live in [cco.config.json](cco.config.json) (`scoring.significance`).
- **Roofline** `pct_peak_*` is *reported-only* — informative, never part of the score (it can't be,
  since the score is champion-relative).

---

## Seed-champion baselines

The seed champions (`champions/<track>/kernel.py`) are the bar each track opens with. The figures
below were measured on the **CCO dev box — NVIDIA RTX 5070 Ti (sm_120 Blackwell), torch 2.8.0+cu128,
triton 3.4.0** via WSL2. **This is the development box, not the canonical SKU** (the canonical GPU is
the pinned SKU used for the scored canonical rerun); treat these as reference, not as the
competition leaderboard. The live leaderboard is the set of `cco-winner-<track>` labels on merged
PRs once the competition is wired.

| Track | Primary size / dtype | Seed-champion baseline (dev box) | Regime |
|---|---|---|---|
| `rms_norm` | 4096×4096, bf16/fp16 | correctness PASS; ~9.6× vs eager PyTorch; ~80% peak BW | memory-bound |
| `matmul` | 2048³, fp16 (fp32 is correctness-only) | PASS; ~91 TFLOPS ≈ 99% cuBLAS | compute-bound |
| `qkv_part_rope` | b2×s4096, bf16 | PASS; ~9.7× vs eager | memory-bound |
| `swiglu_input_quant` | 4096×7168, bf16→fp8 (multi-output) | PASS; ~29.8× vs eager | memory-bound |
| `dsa_forward` | b4×s2048, bf16 (multi-output) | PASS; ~101 TFLOPS, ~18× vs the dense reference | compute-bound |

"vs eager / vs reference" here is the seed champion's headroom over naive PyTorch and is shown only
to characterize each track — **it is not the competition axis.** What miners are scored on is
beating the *current champion* in the table above (which advances as PRs win).

---

## Reproducing a baseline

```bash
# On a CUDA+Triton machine (Linux, or WSL2 on Windows):
cp champions/<track>/kernel.py kernel.py     # put the champion in the scored slot
uv run benchmark.py                          # self-score (seed 42) -> correctness + latency
uv run benchmark.py --blob                   # the full bound blob, incl. identity hashes
```

Numbers should match within ±2% on the same GPU SKU and driver. Larger variance usually means a
thermal / clock difference — lock clocks (`nvidia-smi -lgc/-lmc`) and re-run. On a different SKU the
absolute latencies differ; that is expected and is why a speedup is only compared champion-vs-
challenger on **one** pinned SKU within a single sealed job.

---

## Notes on honesty

We report what the locked harness reproduces. Speedups are measured against the in-repo champion,
never against cuBLAS or a vendor library (for `matmul` the PyTorch reference *is* cuBLAS, so it is
used for correctness only — beating it isn't the game; beating the standing Triton champion is). A
row with no data is intentional; a row with fabricated data is a project-level integrity failure.
