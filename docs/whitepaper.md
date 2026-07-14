# CCO (Cuda-Compute-OSS)

**Verified cheaper GPU compute — from exact matmul to sub-quadratic attention.**

Whitepaper · July 2026

---

## Abstract

CCO is an open-source system for validating and improving GPU compute — today an
arena for **cheaper matrix multiplication**, tomorrow a pipeline for
**sub-quadratic attention kernels**. Contributors submit strategies that compute
the same mathematical result for less cost — lower latency, lower VRAM, fewer
FLOPs — **without giving up accuracy**, and a deterministic, hardware-measured
evaluation harness decides whether the claim is real. No LLM-as-a-judge; no
self-reported numbers on the frontier.

CCO is **not its own network**. It is a whitelisted repository on
[Gittensor](https://gittensor.io/) — Bittensor subnet 74 — which pays TAO for
merged pull requests to recognized open-source projects. Gittensor supplies the
incentive layer (miner registration, PR verification, sybil resistance, payout);
CCO supplies the thing worth paying for: a benchmark where an improvement can be
proven, and a roadmap from today's matmul arena to production attention kernels
that attack the O(n²) wall in Transformer LLMs.

---

## 1. The Problem — Why LLMs Hit a Wall

Modern LLMs are built on the Transformer, whose self-attention computes an
n × n score matrix over n input tokens:

- At **8,000 tokens**: ~64 million score values per attention layer.
- At **128,000 tokens**: ~16 **billion** per layer.
- At **1,000,000 tokens**: ~1 **trillion** per layer — beyond commercially
  available hardware.

Frontier models push the practical limit with engineering workarounds — chunked
attention, sparsity, memory bandwidth — not by changing the underlying O(n²)
math. Meanwhile, sub-quadratic alternatives exist in research but trade away
accuracy, making them unsuitable for precision-critical work.

Underneath attention sits an even more general hot loop: **C = A × B**. Nearly
all numerical computing — training and inference alike — spends its cycles
there. A verified way to compute it more cheaply, at held accuracy, is valuable
far beyond any single model.

CCO attacks the general problem first (Phase 0: matmul, live today) and the
flagship problem on its roadmap (Phases 1–4: attention), with one unbroken rule
across both: **you may only claim an improvement when every cost axis goes down
and accuracy does not.**

---

## 2. Current Status — What Exists Today

We state this plainly rather than over-claim:

| Component | Status |
|---|---|
| Exact O(N³) baseline engine (`matmul/`, in-core + out-of-core tiled, CUDA/MPS) | **Shipped** |
| Subspace strategy + pluggable transform API (`strategy/`) | **Shipped** |
| Deterministic scorer: accuracy · latency · peak VRAM · FLOPs, dominance-gated (`eval/`) | **Shipped** |
| SN74 whitelisting (repository earns TAO for merged PRs) | **Live — 1% emission share** |
| A strategy that beats exact GEMM on the full-rank reference regime | **Open challenge — none yet** |
| Automated PR eval bot on pinned hardware | Planned (§7) |
| Live frontier dashboard + append-only ledger | Planned (§6) |
| Attention-track kernels, drop-in library, commercial API | Roadmap (§8–9) |

The honest headline: the measurement discipline is built and running; the
frontier is untouched. On the reference regime the bundled `rsvd` strategy
**fails by design** — full-rank data has no low-rank structure to exploit, so its
accuracy collapses and its score gates to 0. That is the starting line, not a
defect: the first admitted improvement will mean something precisely because the
bar could not be faked.

---

## 3. How CCO Works on Gittensor (SN74)

CCO does not run its own token, settlement, or registration — Gittensor already
does, audited and in production. The loop:

1. A contributor (an SN74 **miner**) opens a PR — a new transform, strategy, or
   engine improvement — with a self-scored `python -m eval` scorecard.
2. CCO's evaluation harness re-measures the claim on the **pinned reference
   regime** (§6) and produces a deterministic verdict label:
   `eval:S` … `eval:L` for a verified frontier improvement, `eval:none` for
   correct-but-not-better, `eval:REJECT` for accuracy loss.
3. A maintainer merges PRs on their engineering merits; only the eval label
   carries score weight.
4. **SN74 validators** read the merged PR and its label through Gittensor's
   per-repository configuration (`master_repositories.json`), and the
   contributor earns TAO from CCO's emission share automatically.

| Role | Who/what | Key output |
|---|---|---|
| Contributor | Any SN74 miner with a GitHub PAT | PR + scorecard |
| Evaluation harness | CCO's `eval/` (maintainer-operated) | Deterministic `eval:*` label |
| Incentive layer | Gittensor SN74 validators | TAO payout per label multiplier |
| Maintainers | CCO team | Merges, regime pins, ledger, dashboard |

Inheriting the incentive layer is a feature, not a compromise: miner
registration, PAT-based identity verification, credibility gates, PR-spam
collateral, and payout settlement are Gittensor's responsibility — CCO's whole
engineering budget goes into the part nobody else provides: **a benchmark that
cannot be gamed.**

> **Note on rewards.** Gittensor is not winner-take-all. Every merged PR earns
> according to its eval label's multiplier within CCO's emission share. The
> competitive ratchet survives in the *frontier rule*: a scoring label is only
> awarded for a verified improvement over the current best — matching or
> re-deriving the frontier scores `eval:none`.

---

## 4. What Contributors Do

### 4.1 The core task (Phase 0 — live today)

Submit a **strategy** that computes `C = A × B` for less compute cost at held
accuracy. Most contributions are a new **transform** — the pluggable basis of
the subspace strategy:

```python
from strategy.transforms import Transform, register_transform

class MyTransform(Transform):
    name = "mine"
    def basis(self, n, m, backend, dtype, A=None, B=None):
        Q = ...  # (n, m), orthonormal columns, on backend.xp
        return Q

register_transform("mine", MyTransform)
```

Whole new strategies — including **exact-but-cheaper** algorithms (e.g.
Strassen-class FLOP reductions), sparse methods, or mixed-precision schemes —
are equally welcome and face the same gate.

### 4.2 Open-source requirement

CCO is a public MIT repository; every submission is a public PR. This is what
lets the harness re-run any claim from source, and lets every contributor study
and improve on the current frontier — public, evolutionary competition rather
than private, parallel guessing.

### 4.3 What a submission contains

- **Code**: the transform/strategy, registered and selectable by name.
- **Scorecard**: the machine-readable output of
  `python -m eval --n 8192 --pairs 3 --transforms mine --json`, produced on
  the contributor's GPU.
- **Honest FLOP accounting**: `basis_flops()` for any non-negligible basis
  construction cost (verified by the harness, §7).

---

## 5. The Evaluation Engine

Scoring is fully deterministic — a function of measurements, so independent
re-runs converge on the same verdict.

### 5.1 Tier 1 — Accuracy gate (pass/fail)

The exact product `C` and the strategy's `Ĉ` are computed **on identical
inputs** in one pass; accuracy is compared in float64:

```
accuracy = max(0, 1 − ‖C − Ĉ‖_F / ‖C‖_F)        # bounded to [0, 1]
```

Accuracy below the regime's floor → **score 0**, label `eval:REJECT`. A fast
wrong answer is a different, worse answer. (The attention track, Phases 1–4,
uses MSE against the exact attention output plus a full-model compounding
check — same principle, layer-appropriate metric.)

### 5.2 Tier 2 — The dominance gate

A strategy is admitted as an **improvement** only if, versus the exact baseline
on the same inputs, **all** cost axes strictly reduce:

- wall-clock latency (GPU-synchronized), **and**
- peak incremental VRAM (worst case over couples), **and**
- FLOP count (including basis construction).

One regressing axis disqualifies. No averaging a win on one axis against a loss
on another; no aggregating sub-threshold gains across regimes. The
time-complexity claim is additionally checked empirically: latency is fit to
`N^p` over a size sweep, and a claimed sub-cubic method must show it.

### 5.3 Tier 3 — Composite score & label

Admitted strategies are ranked by:

```
score = accuracy × (1 / Peak_VRAM) × (1 / Latency)
```

The verdict label is a deterministic bucket of the verified improvement over
the current frontier (thresholds governance-tunable, mirroring proven SN74
practice):

| label | verified improvement over frontier |
|---|---|
| `eval:L` | ≥ 25% |
| `eval:M` | ≥ 10% |
| `eval:S` | ≥ 2% (the significance floor — below it is noise, label `eval:none`) |
| `eval:BASELINE` | first verified entry on a new track |
| `eval:none` | correct, but no verified frontier improvement |
| `eval:REJECT` | fails the accuracy gate |

Sub-2% gains are never aggregated across tracks or sizes to manufacture a
scoring label.

---

## 6. Reference Regime & Live Frontier

*The credibility core of CCO: fixed hardware, fixed workload, pinned baseline —
so every merged PR is one comparable point on one public curve.*

### 6.1 The pins

| Pin | Value |
|---|---|
| **GPU** | One fixed evaluation device (reference: NVIDIA RTX 5090), same physical class for every eval; clocks recorded per run |
| **Workload** | `N = 8192`, fp32, **full-rank** random couples, 3 pairs, per-eval logged seeds |
| **Baseline** | Exact `torch.matmul` product, pinned PyTorch + CUDA versions, measured in the same run on the same box |
| **Environment** | Pinned container image; any change to a pin starts a new, clearly-marked frontier era |

Same-run, same-box measurement means box-to-box hardware variance cancels: the
scored quantity is the **delta versus exact on identical inputs**, not an
absolute number that depends on silicon lottery.

### 6.2 Tracks

The frontier is per-track. Each track is one data regime with its own floor and
its own ledger:

| track | data | accuracy floor | status |
|---|---|---|---|
| `full-rank` (reference) | random, no exploitable structure | 0.80 | **open — unbeaten** |
| `low-rank` | rank ≪ N couples | 0.95 | open |
| `decaying-spectrum` | polynomially decaying singular values | 0.90 | planned |
| `attention-shaped` | rectangular `softmax(QKᵀ/√d)·V` at fixed model shapes | phase-dependent (§8) | planned — the bridge to Phases 1–4 |

A gain on one track never transfers to another; regressions on guarded tracks
are labeled explicitly (`regression-<track>`).

### 6.3 The frontier ledger and dashboard

Every evaluation appends one line to an **append-only, GitHub-timestamped
ledger** (`ledger.jsonl`): `(date, PR, author, commit, track, seed, accuracy,
latency, peak VRAM, FLOP ratio, verdict)` — auditable line-by-line against the
per-run logs. Nothing on the frontier is ever edited in place.

A public **dashboard** (fed by `dashboard/data.json` on the
`bot/dashboard-state` branch) renders the ledger: the current frontier per
track, the exact-baseline anchor, and the **optimization journey** — one point
per merged frontier PR, from the baseline to the current best. The chart is the
project's progress report; if the curve doesn't move, nothing is claimed.

### 6.4 Reproduce it yourself

```bash
git clone https://github.com/zeokin/Cuda-Compute-OSS && cd Cuda-Compute-OSS
python -m eval --n 8192 --pairs 3                 # reference regime scorecard
python -m eval --n 8192 --pairs 3 --json          # machine-readable
python tests/test_correctness.py                   # the baseline's own gates
```

Anyone with a CUDA/MPS GPU can re-run any scorecard from source. The frontier
is trustworthy because it is *reproducible*, not because we assert it.

---

## 7. Anti-Gaming Mechanisms

Real money attracts score-optimizers. Design status, stated honestly:

| Attack vector | Defense | Status |
|---|---|---|
| Overfitting the known eval inputs | **Fresh random seed per evaluation**, logged for reproducibility (fixed seeds are for contributor self-runs only) | **Planned — priority #1** |
| Claiming performance without running the code | Maintainer-operated eval bot independently re-builds and re-runs every scoring PR on the pinned device | Planned |
| Understating basis/setup FLOPs | Empirical FLOP verification (profiler counts) against the declared `basis_flops()` | Planned |
| Submitting exact O(N³) re-labeled as sub-cubic | Empirical `N^p` scaling fit over a size sweep | **Shipped** (`--sweep`) |
| Accuracy traded for speed | Hard accuracy floor; score gates to 0 | **Shipped** |
| Cherry-picked inputs / regimes | Fixed reference regime; per-track frontiers; losses published | **Shipped** (policy + harness) |
| Copying an earlier PR to farm credit | Diff-containment + structural copycat detection, graduated strikes → block (proven SN74 pattern) | Planned |
| Micro-win aggregation | 2% significance floor per track; no cross-track summation | **Shipped** (policy), enforced by label function |

Identity, sybil resistance, PR-spam collateral, and credibility gating are
provided by the Gittensor layer and are already live for every CCO PR.

---

## 8. Incremental Difficulty Roadmap

Phase transitions are triggered by achievement, not calendar.

| Phase | Target | Bar | Status |
|---|---|---|---|
| **0 — Matmul arena** | `C = A × B` tracks (§6.2) on the CCO harness | Dominance gate at per-track accuracy floors | **Live** |
| **1 — Foundation** | Attention at ≤ 32K context | MSE < 0.01 vs exact attention; adapting open-source kernels (FlashAttention variants) qualifies | Roadmap |
| **2 — Log-linear** | ≤ 128K context | MSE < 0.005; requires genuinely novel hybrid/spectral/low-rank kernels + scaling proof | Roadmap |
| **3 — Million-token** | ≤ 1M context | MSE < 0.001; production-grade sub-quadratic VRAM scaling, Triton/CUDA hardware tuning | Roadmap |
| **4 — Convergence** | Arbitrary context | MSE < 0.0005; optional weight co-optimization (LoRA patches) to recover residual error | Roadmap |

Phase 0 keeps short-term rewards accessible (any skilled GPU/numerics engineer
can attack a track today) while the attention phases hold the long-term prize.
The periodic **full-model spot check** (a fact injected deep in a long sequence
must survive the kernel across all layers) guards Phases 1+ against error
compounding.

---

## 9. The Offering

1. **Now — the open harness.** A rigorous, reproducible arena for approximate
   and exact matmul strategies: the exact engine, the pluggable strategy API,
   and the dominance-gated scorer. Its users are contributors and researchers.
2. **Next — the drop-in library.** `pip install cco`; `from cco import matmul`
   as a torch-compatible drop-in backed by the current per-track champion, with
   the accuracy floor as an explicit, user-chosen parameter.
3. **Roadmap — champion attention kernels.** Per Phase 1+: an open-source
   library of hardware-validated sub-quadratic attention kernels, publishable
   as a drop-in attention module for Transformer inference stacks — and, once
   real, a commercial API wrapping the champion kernel (compatibility testing,
   regression checks, versioning) for inference providers and long-context
   enterprise workloads.

Nothing in tier 3 is sold before it exists. The dashboard (§6.3) is the public
record of the distance between here and there.

---

## 10. Economics on SN74

CCO's economics run on Gittensor today — no token launch, no treasury to
bootstrap, no external capital dependency:

- **Contributor revenue**: merged PRs earn TAO from CCO's emission share,
  weighted by the deterministic `eval:*` label multipliers. Verified frontier
  work pays multiples of routine work; unverified work pays nothing extra.
- **Project revenue**: a `maintainer_cut` of the repository's emissions funds
  eval hardware, the bot, and maintenance.
- **The flywheel**: verified frontier progress → a moving public dashboard →
  a credible case (made by PR to the subnet's weight registry) for a larger
  emission share → more and stronger contributors → faster frontier progress.
  This exact loop has been demonstrated on SN74: the top repository grew its
  emission share to 34% by shipping a verified, bot-enforced frontier. CCO's
  current 1% is the starting position, not the ceiling.
- **Long term**: if and when champion attention kernels exist (Phases 1–4),
  API/library monetization (§9.3) adds a revenue stream *outside* the subnet —
  at which point reinvesting it into contributor incentives becomes a
  governance decision made in the open.

---

## 11. Why This Succeeds Where Others Have Failed

| Challenge | Common failure | CCO's answer |
|---|---|---|
| LLM-as-a-judge bias | Judged systems inherit judge hallucinations | Pure measurement: Frobenius/MSE accuracy, VRAM, latency, FLOPs on pinned hardware |
| One-time solution kills competition | A single breakthrough ends the race | Per-track frontiers, tightening floors, phase ladder to 1M-token attention — the bar never relaxes |
| Too hard for new contributors | PhD-only from day one | Phase 0 tracks are attackable with existing numerical-linear-algebra literature; difficulty ratchets by achievement |
| No real incentive plumbing | Grants or goodwill run out | Gittensor pays per merged PR **today**; no settlement system to build or fund |
| Scam-vulnerable scoring | Self-reported numbers, judge gaming | Deterministic labels, re-execution from source, seed randomization, scaling proofs, copycat detection |
| Marketing outruns the code | Vision docs describe systems that don't exist | §2 states exactly what is shipped vs planned; the ledger and dashboard make progress — or its absence — public |

---

## 12. Conclusion

The quadratic cost of attention — and beneath it, the cubic cost of the matrix
product — is the defining bottleneck of modern AI economics. CCO's bet is that
the way through is not one paper but a **standing, verified, incentivized
competition**: an evaluation harness nobody can game, an incentive stream that
already pays (Gittensor SN74), one fixed regime where every improvement is a
public point on a public curve, and a difficulty ladder that starts at matmul
strategies anyone can attempt this week and ends at million-token attention.

The frontier is open. The scorer is running. The first verified point on the
curve is the whole game.

---

*Technical parameters — accuracy floors, label thresholds, track definitions,
regime pins — are governance-tunable and may change as the evaluation
infrastructure (§6–7) evolves. CCO is a whitelisted repository on Bittensor
subnet 74 (Gittensor); it is not a separate network, token, or investment
product. Nothing herein constitutes financial advice.*
