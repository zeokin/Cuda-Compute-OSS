# Eval — scoring the smart (subspace) strategy

An evaluation harness that measures how good the **smart (subspace)** method in
[`strategy/`](../strategy/) is, relative to **normal (exact)** computing, and
turns it into a single comparable **score** per transform strategy.

## What it does

```
1. generate  P random couples (A_i, B_i), each N × N
2. normal    C_i = A_i @ B_i          (exact, streamed)
3. smart     Ĉ_i = subspace(A_i, B_i) (per transform: rsvd / your own / …)
4. estimate  accuracy · latency · peak VRAM · FLOP complexity
5. score     accuracy × (1/Peak_VRAM) × (1/Latency)   (0 unless it beats exact
             on accuracy AND every cost axis — the dominance rule)
```

The exact products are computed **once** and reused for every transform, so all
transforms are judged on identical inputs.

## The metrics

**Accuracy** — the Relative Frobenius Norm Error folded into a bounded `[0, 1]`
score so it plugs safely into the product formula (never negative, never blows
up):

```
Accuracy = max(0, 1 − ‖C − Ĉ‖_F / ‖C‖_F)
```

**Latency** — wall-clock seconds of the smart multiply (GPU-synchronized),
averaged across couples.

**Peak VRAM** ("pick of VRAM") — peak *incremental* device memory during the
multiply ([`memory.py`](memory.py)). On CUDA this is exact, from the torch
caching allocator (`reset_peak_memory_stats` + `max_memory_allocated`); on MPS
it is sampled from `torch.mps.current_allocated_memory`.

**Time complexity** — reported analytically (normal `O(N³)`, smart `O(N²·M)`)
and, with `--sweep`, fitted empirically to `latency ~ N^p` in log-log space.

**Score** — rewards accurate, memory-light, fast strategies, but only among
strategies admitted as an **improvement** over exact. It is hard-gated to 0
unless accuracy clears the floor (`--min-accuracy`, default **0.8**) **and** the
strategy dominates the exact baseline on every cost axis — latency, peak VRAM
**and** FLOP count all below exact (the dominance rule in
[BENCHMARKS.md](../BENCHMARKS.md)). So "fast but wrong", or accurate but slower /
heavier than exact, cannot win:

```
score = Accuracy × (1 / Peak_VRAM) × (1 / Latency)
      → 0 unless Accuracy ≥ floor AND latency, VRAM, FLOPs all below exact
```

Peak_VRAM is expressed in `--vram-unit` (default GiB) so the number stays
readable; the score is a **relative ranking** metric across transforms measured
under the same units.

## Use it — CLI

```bash
# Reference regime: 12000, full-rank, 3 couples (all defaults), + scaling fit.
# Subspace can't approximate full-rank -> accuracy ~0; this is the honest baseline.
# (--rank-m holds M fixed for the sweep so it isolates the ~N² term.)
python -m eval --n 12000 --pairs 3 --rank-m 128 --sweep 512,1024,2048

# The strategy's happy path — compressible (low-rank) data, where it wins:
python -m eval --n 12000 --pairs 3 --fill lowrank --data-rank 16

# The accuracy floor defaults to 0.8; override it (or 0 to disable), emit JSON:
python -m eval --n 12000 --min-accuracy 0.9 --json
```

Compute is **GPU-only** (PyTorch on CUDA/MPS); with no GPU the CLI prints a clear
error.

Key flags: `--n`, `--pairs`, `--dtype {fp16,fp32,fp64}`, `--rank-m M`,
`--fill {random,lowrank,iota}`, `--data-rank`, `--transforms rsvd`,
`--min-accuracy`, `--vram-unit {bytes,mib,gib}`, `--sweep`, `--device`, `--json`.

## Use it — Python API

```python
from eval import EvalConfig, evaluate, estimate_scaling

# reference regime: 12000, full-rank (fill defaults to "random")
out = evaluate(EvalConfig(n=12000, pairs=3, rank_m=128))
print(out["best"], out["ranking"])
print(out["transforms"]["rsvd"])   # accuracy, latency_s, peak_vram_bytes, score

fit = estimate_scaling([512, 1024, 2048], EvalConfig(rank_m=128))
print(fit["fitted_exponent_p"])    # empirical p in latency ~ N^p
```

## Interpreting results

On the reference **full-rank** `12000` data *no* subspace of `M ≪ N` can
approximate the product, so every transform's accuracy collapses to ≈ 0 (and any
accuracy floor zeroes the score) — the honest baseline the strategy is **not**
for. On **low-rank** data (`--fill lowrank`) the data-aware `rsvd` transform
reconstructs almost exactly (accuracy ≈ 1) and dominates the score, because it
builds its subspace from A and B themselves. Even then the score is non-zero
only when the strategy also beats exact on cost (latency, VRAM and FLOPs); an
accurate strategy that is slower or heavier than exact is not an improvement and
is scored 0.

## Layout

```
eval/
  metrics.py    accuracy (bounded Frobenius), the score formula + accuracy gate
  memory.py     MemoryProbe — peak GPU VRAM (CUDA exact / MPS sampled)
  evaluator.py  generate couples, run normal+smart, collect metrics, fit scaling
  cli.py        python -m eval
  tests/        unit tests (run on CPU)
```

## Test

```bash
python eval/tests/test_eval.py        # or: python -m pytest eval/tests -q
```
