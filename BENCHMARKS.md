# Benchmarks

Reference setup: **`12000 × 12000`** matrices, **full-rank** (random) data, `fp32`,
on an **A100 (80 GB)** GPU via PyTorch. This is the default the scorer
runs, and the hardest, most honest case — there is no low-rank structure to
exploit.

CCO compares two ways to compute `C = A × B` on the **same inputs**:

- **general (normal / exact)** — the `O(N³)` engine in [`matmul/`](matmul/) /
  `strategy.multiply_exact`. This is the reference; its answer *is* the truth.
- **suggested (smart / subspace)** — an approximate strategy in
  [`strategy/`](strategy/) that trades exactness for lower compute cost.

Every number on this page is produced by the scorer in [`eval/`](eval/), so
anyone can reproduce it with one command. Nothing here is hand-tuned or
hand-copied.

---

## How a number is produced

One `python -m eval …` run does all four measurements in a single pass:

1. **Generate** `--pairs` random couples `(Aᵢ, Bᵢ)`, each `N × N`, from a fixed
   seed. The exact products are computed **once** and reused for every strategy,
   so all strategies see identical inputs.
2. **accuracy** — the exact `C` and the smart `Ĉ` are compared in float64:
   ```
   accuracy = max(0, 1 − ‖C − Ĉ‖_F / ‖C‖_F)          # bounded to [0, 1]
   ```
   This same bounded accuracy is what the correctness gate uses (below).
3. **time complexity** — reported two ways: the analytic order (`normal O(N³)`,
   `smart O(N²M)`) and the FLOP ratio the strategy actually performs, plus an
   **empirical** exponent `p` fit from `latency ~ N^p` over `--sweep N₁,N₂,…`
   (rank `M` held fixed so `M` doesn't grow with `N`).
4. **latency** — wall-clock seconds of the multiply, with the device
   synchronized before and after so async GPU work is fully counted. Reported as
   the mean over the couples.
5. **VRAM usage** — peak *incremental* GPU memory for the whole multiply
   ([`eval/memory.py`](eval/memory.py)): on CUDA it is exact, from the torch
   caching allocator (`reset_peak_memory_stats` + `max_memory_allocated`); on MPS
   it is sampled from `torch.mps.current_allocated_memory`. Worst case over the
   couples.

The four are folded into one ranking score, hard-gated by correctness **and** by
the dominance rule below — a strategy scores non-zero only when it is admitted as
an improvement over exact:

```
score = accuracy × (1 / Peak_VRAM) × (1 / Latency)
# 0 unless accuracy ≥ floor AND latency, VRAM and FLOPs are all below exact
```

Reproduce the reference comparison on your GPU with:

```bash
python -m eval --n 12000 --pairs 3 --transforms rsvd \
               --rank-m 128 --sweep 512,1024,2048
```

---

## general vs. suggested — the comparison

The comparison is always **per regime** (dtype, matrix content, size), measured
on your GPU. The reference regime is `N=12000`, 3 couples, `fp32`, **full-rank**
(random) data, subspace `M = N//8 = 1500`:

| aspect              | general (exact) | suggested (rsvd) | suggested better? |
|---------------------|-----------------|------------------|-------------------|
| accuracy            | 1.0000 (truth)  | **≈ 0**          | **no** — collapses |
| time complexity     | `O(N³)`         | `O(N²M)`, ~4× fewer FLOPs, fitted `N^p` | yes (FLOPs) |
| latency             | _your GPU_      | _your GPU_       | measure           |
| VRAM usage          | _your GPU_      | _your GPU_       | measure           |

Accuracy and the FLOP/complexity columns are device-independent; latency and VRAM
must be measured on the target GPU and pasted in.

### The improvement rule

> A suggested method is admitted as an **improvement** over the general method
> only when, on the same regime, **all** of these hold at once:
>
> - error (`1 − accuracy`) does **not** increase, **and**
> - time complexity **reduces**, **and**
> - latency **reduces**, **and**
> - VRAM usage **reduces**.
>
> If *every* item reduces (with accuracy held), we admit the improvement. If any
> one regresses, we do **not** — no averaging a win on one axis against a loss on
> another.

We express accuracy as **error** (`1 − accuracy`) precisely so all four axes read
"lower is better" and the rule is a clean dominance check.

**Reading the verdict — full-rank is hard on purpose.** On full-rank `12000` data
a subspace of `M ≪ N` cannot represent the product: the error is ~100%, accuracy
`≈ 0`, and the accuracy floor forces the **score to 0**. Fewer FLOPs do not help
when the answer is wrong. So on the reference regime the honest result is: **the
reference subspace strategy does not beat exact — use exact.** A real improvement
here means a transform that captures structure the reference `rsvd` basis misses,
lifting accuracy *while* cutting cost.

### Where the suggested method wins

The subspace method pays off only when the data is **compressible** (low-rank /
smooth) — then `M ≪ N` captures it and accuracy holds:

- **Compressible data** (low-rank / smooth) at `M ≪ N` — accuracy holds and
  `O(N²M) ≪ O(N³)`. Show it explicitly:
  ```bash
  python -m eval --n 12000 --pairs 3 --fill lowrank --data-rank 16 --transforms rsvd
  ```
- **out-of-core scale** — where the exact `O(N³)` product does not fit in GPU
  memory at all, so a smaller subspace multiply is the *only* thing that runs.

Those numbers must be produced on the target hardware and pasted into the PR;
they are not asserted here.

---

## Notes on honesty

- **Same inputs, one run.** Accuracy and latency for a strategy always come from
  the *same* couples at the *same* dtype. We never pair an accuracy number from
  one run with a latency number from another.
- **The reference is the truth, and it is tested.** The exact engine is gated by
  [`tests/`](tests/) (ragged tiles, fp16 accumulation) on the GPU. A speed claim
  against an unverified baseline is meaningless, so we don't make one.
- **Peak, not average, memory.** VRAM is the *peak* of the PyTorch caching
  allocator during the multiply — every transient tensor and workspace that goes
  through PyTorch's allocator, not smoothed away. Allocations a library makes
  *outside* PyTorch's allocator (raw `cudaMalloc` workspace) are not counted, so
  treat the number as a lower bound on true device use.
- **Wall-clock, fully synchronized.** GPU latency counts async work; we
  synchronize before timing and after. FLOP-count wins (`O(·)`) are reported
  separately from measured latency, because fewer FLOPs is *not* the same as less
  wall-clock time — at small `N` the two often disagree.
- **We publish losses.** A regime where the suggested method loses is shown, not
  hidden. A strategy that only wins on cherry-picked data is documented as
  exactly that.
- **No aggregation to manufacture a headline.** Sub-threshold or single-regime
  gains are not summed across sizes to claim a general win. Each claim stands on
  one regime, with its command line.
- **Numbers over narrative.** If a description and its scorecard disagree, the
  scorecard is authoritative.

---

## Reproduce everything

All of this runs on a GPU (CUDA/MPS) via PyTorch.

```bash
# the reference comparison on this page (12000, full-rank)
python -m eval --n 12000 --pairs 3 --transforms rsvd \
               --rank-m 128 --sweep 512,1024,2048

# machine-readable numbers (for a PR scorecard)
python -m eval --n 12000 --pairs 3 --transforms rsvd --json

# the correctness gates (skip if no GPU is present)
python tests/test_correctness.py
python eval/tests/test_eval.py
python strategy/tests/test_subspace.py
```

Always report the GPU and dtype you used.
