# Contributing to CCO

CCO accepts two public PR lanes. The current score-bearing track is **matmul**:

- **`fix` / `bug` PRs** correct mistakes in the repository. They must pass the
  CPU-safe validation path, but they do **not** need a GPU scorecard.
- **`feat` / `strategy` PRs** claim a matmul improvement. They must pass the
  same CPU-safe validation **and** include a GPU scorecard from the shared
  scorer.

This document defines both lanes: **the one rule**, **what is score-bearing
today**, **local validation**, **submit**.

## Where Gittensor Fits

This repository is part of work being built with help from **Gittensor**. That
relationship should be understandable from GitHub alone:

- Gittensor helps power the project and gives the work a broader network context.
- This GitHub repository is still the public source of truth for code, docs,
  benchmarks, and pull requests.
- If you want to contribute, you do not need to be in Discord or already know
  SN74. You can participate directly here by reading the docs, running the
  scorer, and opening a PR.

---

## The one rule

> **You may only claim an improvement when every cost axis goes down and accuracy
> does not.**

Cost is `time complexity`, `latency`, and `VRAM usage`. Accuracy is the bounded
Frobenius score. A change that makes the multiply faster or smaller **by making
it less accurate is not an improvement** — it is a different, worse answer, and
CCO scores it as one (often `0`, via the accuracy floor).

Concretely, a submission is admitted as an improvement only if, against the
current frontier on the **same inputs**:

- error (`1 − accuracy`) does **not** increase, **and**
- `latency`, `VRAM`, and the empirical time-complexity exponent **all** decrease.

If any cost axis regresses, or accuracy drops, it is not an improvement. No
exceptions, no averaging a loss on one axis against a win on another. See
[BENCHMARKS.md](BENCHMARKS.md) for the precise dominance rule.

Everything else in this file is just how to *demonstrate* that you followed the
one rule.

---

## What you actually change

Most score-bearing contributions today are a new **transform** — the pluggable
basis that defines the subspace the smart strategy compresses into. You add a
class in [`strategy/transforms.py`](strategy/transforms.py) and register it:

```python
from strategy.transforms import Transform, register_transform

class MyTransform(Transform):
    name = "mine"
    def basis(self, n, m, backend, dtype, A=None, B=None):
        # return an (n, m) array on backend.xp with ORTHONORMAL columns
        Q = ...
        return Q

register_transform("mine", MyTransform)
```

That is enough to be scored: `--transform mine`. Bigger contributions (a new
compression scheme, a better exact tile schedule in `matmul/`) are welcome too.

If your PR is fixing incorrect behavior rather than claiming a new measured
improvement, submit it through the `fix` lane instead of inventing a scorecard.

### Current tracks

| track | status | what contributors should do |
|---|---|---|
| `matmul` | Live score-bearing track | Submit `feat` / `strategy` PRs with `python -m eval ...` scorecards. This is the only public track that currently enters the GPU queue. |
| `attention` | Prototype / roadmap track | Use `attention/` for local experiments and maintainer-reviewed prototype work only. It is not a scored miner track yet, and attention PRs must not claim a `python -m eval` matmul scorecard as proof. |

The future attention track will need a separate exact baseline, quality metric,
validator command, bot routing, and dashboard result model. Until those pieces
exist, attention work is useful research infrastructure, not an admitted
score-bearing improvement.

---

## Two PR lanes

### `fix` / `bug` lane

Use this lane when the PR corrects repository behavior:

- wrong math
- validation gaps
- crashes / OOM routing bugs
- test coverage for an existing bug
- docs-only clarifications

What is required:

- title or PR body must declare `fix` / `bug`
- CPU-safe validation must pass
- no GPU scorecard is required
- the PR does **not** enter the GPU queue

### `feat` / `strategy` lane

Use this lane when the PR claims a matmul improvement worth measuring:

- new transform
- new approximation strategy
- performance feature
- algorithmic change that claims lower cost at held accuracy

What is required:

- title or PR body must declare `feat` / `strategy`
- CPU-safe validation must pass
- a filled matmul GPU scorecard from `python -m eval ...`
- the PR enters the sequential GPU queue after non-GPU triage

---

## What's open, what's protected

| zone | paths | policy |
|---|---|---|
| **Open — the main event** | [`strategy/transforms.py`](strategy/transforms.py) (new `Transform` classes), new strategy modules under `strategy/` | The designed hook — `register_transform()` exists for exactly this. Self-scored, verified by the scorer below. |
| **Open — engine performance** | [`matmul/`](matmul/), `strategy/backend.py`, `strategy/subspace.py` | Tiling, streaming, dtype/precision paths, platform fixes — all welcome, measured the same way. |
| **Open — prototype research** | [`attention/`](attention/) | Local attention-shaped experiments. Useful for future track design, but not score-bearing until the evaluator and bot support an official attention track. |
| **Open — accompanying** | `tests/`, `strategy/tests/`, `examples/` | Welcome alongside a code change. Test-only PRs score `0` by design — they don't demonstrate a cost improvement. |
| **Protected — maintainer-owned** | [`eval/`](eval/) (the scorer, the bot, the ledger), [`docs/`](docs/), [`.github/`](.github/), `dashboard/` | The scoring machinery itself. See [`.github/CODEOWNERS`](.github/CODEOWNERS) — a PR touching these paths is held for maintainer review regardless of what else it does. Changing the scorer is not how you win on it. |

Not sure which zone a change falls into? Open the PR anyway — CODEOWNERS
review routes it correctly.

If a non-maintainer PR touches `eval/`, `docs/`, `.github/`, or
`dashboard/`, the sensitive-paths guard fails the PR. That is expected:
those files define the rules, automation, or public score feed. A legitimate
change there should be split into a maintainer-reviewed PR instead of bundled
with a miner scoring submission.

---

## Local validation

CCO uses **uv**. Install the CPU-safe contributor environment first:

```bash
uv sync --extra test
```

**Step 0 — a fast, no-GPU sanity check.** Before touching a GPU at all:

```bash
uv run python -m strategy.smoke
```

This exercises every registered transform's basis on a tiny matrix (shape,
orthonormality, no NaN/Inf) in under a second, on any machine — CPU included.
It's a pre-flight, not a score: passing it proves your transform doesn't
crash, nothing more. The real scorecard always comes from the GPU commands
below.

Every PR lane starts with the same CPU-safe validation:

```bash
uv sync --extra test
uv run python -m strategy.smoke
uv run --extra test python -m pytest tests/ strategy/tests/ eval/tests/ -v
```

That is enough for the `fix` lane.

Before you open a matmul `feat` PR, also run the scorer. It generates random
couples, multiplies them with the **normal (exact)** engine and your **smart**
strategy on the *identical* inputs, and prints one scorecard.

CCO computes on a **GPU** (CUDA/MPS) via PyTorch — score on a GPU machine
(reference: RTX 5090). The reference regime is **`12000`, full-rank**
(random) data, which is `eval`'s default.

```bash
# score your transform on the reference regime (12000, full-rank, 3 couples)
uv sync --extra test --extra gpu
uv run python -m eval --n 12000 --pairs 3 --transforms mine,rsvd

# fit the empirical time complexity O(N^p); pass --rank-m to hold M fixed (~N²),
# omit it to let M = N//8 grow with N (~N³)
uv run python -m eval --transforms mine --rank-m 128 --sweep 512,1024,2048

# machine-readable, for pasting exact numbers
uv run python -m eval --n 12000 --pairs 3 --transforms mine --json

# if your strategy targets compressible data, show that regime too (and say so):
uv run python -m eval --n 12000 --pairs 3 --fill lowrank --data-rank 16 --transforms mine
```

Rules for an honest local matmul `feat` score:

- Score on **unseen** couples from the same distribution — never special-case the
  seeds, sizes, or matrices the harness uses.
- Report accuracy and latency from the **same run** at the **same dtype**.
- Use the peak-VRAM number the scorer measures; do not exclude scratch memory.
- Name the GPU (and dtype) you measured on — results depend on the device.

Optional attention prototype check:

```bash
uv run --extra test python -m pytest tests/test_attention_playground.py -q
```

This verifies the local attention playground only. It is not a public scorecard
and does not put a PR into the GPU queue.

---

## Submit

1. **Fork & branch.** One strategy (or one focused change) per PR.
2. **Keep it standalone.** `matmul/`, `strategy/`, `attention/`, and `eval/` do
   not import each other except where they already do; don't add cross-coupling.
3. **Choose the lane explicitly.** Use `fix:` / `bug:` or `feat:` / `strategy:`
   in the PR title, or check the matching box in the PR template.
4. **Green CPU-safe validation.** `strategy.smoke` (if relevant) and
   `pytest tests/ strategy/tests/ eval/tests/ -v` must pass.
5. **Open the PR**. The PR template
   ([`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md))
   now has both lanes. `feat` PRs must fill in the GPU scorecard section;
   `fix` PRs should fill in the validation section instead.
6. **Your own work.** A PR that substantially reproduces an earlier PR's diff
   is detected automatically and blocked (see
   [`.github/COPYCATS.md`](.github/COPYCATS.md)) — an independently-arrived-at
   similar solution is fine, a copy is not.

If you are coming in through Gittensor, contribute through the same GitHub path
as everyone else: code in a branch, benchmark locally, and submit a PR with
reproducible numbers.

### PR description format

Every PR description must declare exactly one lane:

```markdown
## PR kind

- [x] fix
- [ ] feat
```

or:

```markdown
## PR kind

- [ ] fix
- [x] feat
```

For `feat` PRs, include the scorecard:

```markdown
## Summary

<what the strategy does, why it is cheaper, and the regime it targets>

## GPU Result   (N=12000, full-rank, RTX 5090, fp32)

| metric          | value          |
|-----------------|----------------|
| accuracy        | 0.83           |
| time complexity | O(N²M) ~ N^2.1 |
| latency         | 41.3 ms        |
| VRAM usage      | 12.4 MiB       |
```

- **accuracy** — bounded Frobenius accuracy in `[0,1]` from the scorer. On the
  full-rank reference regime a blind subspace basis lands near `0`; a real
  improvement means finding structure that pushes it up while cutting cost.
- **time complexity** — the analytic `O(·)` and the fitted `N^p` from `--sweep`.
- **latency** — mean wall-clock ms of the smart multiply, GPU-synchronized.
- **VRAM usage** — peak incremental GPU memory during the multiply.

Paste the raw scorecard (or `--json` output) and name the device/dtype you
measured on, so a reviewer can reproduce your numbers exactly.

For `fix` PRs, replace the scorecard with the concrete bug description and the
CPU-safe commands you ran.

### Review & merge

A maintainer reproduces your scorecard on the reference setup, checks the
correctness gates and the one rule, and merges if your matmul strategy is a
genuine improvement (or a useful strategy that documents its trade-off
honestly). If the scorecard can't be reproduced, the PR goes back for evidence
— not rejected for disagreeing with the prose.

The PR bot runs continuously for non-GPU triage. On each PR event and on a
15-minute schedule it checks drafts, blocked contributors, copycat overlap, PR
lane declaration, and scorecard presence when the lane is `feat`.

- `fix` / `bug` / docs PRs are labeled `status:ready-non-gpu` after triage.
  They do not enter the GPU queue.
- matmul `feat` / `strategy` PRs that pass those gates get `status:queued-gpu`
  and appear in `dashboard/data.json` on the `bot/dashboard-state` branch in
  oldest-PR-first order.
- attention prototype PRs do not enter the GPU queue until the official
  attention track is implemented.

The dashboard UI itself is expected to live in a separate private repository,
so `main` stays protected while the bot publishes queue/result data to that
dedicated state branch.

GPU evaluation is intentionally batched. The bot can run all day, but GPU tests
should run sequentially during one or two maintainer-controlled windows per day.
That keeps GPU rental predictable while preserving a public queue of what will
be tested next.

Maintainers can preview the next GPU batch without renting hardware:

```bash
git show origin/bot/dashboard-state:dashboard/data.json > dashboard/data.json
uv run --extra test python -m eval.gpu_batch --limit 3
```

When a GPU is available, run the same queue sequentially:

```bash
git show origin/bot/dashboard-state:dashboard/data.json > dashboard/data.json
uv run --extra test python -m eval.gpu_batch --limit 3 --run --clean
```

By default the GPU scorer omits `--seed`, so every official run sees fresh
unseen matrices. Pass `--seed <n>` only to reproduce a prior result.

**Scoring is moving to automated verdict labels** (`eval:S`, `eval:M`,
`eval:L`, plus `eval:BASELINE`, `eval:none`, `eval:REJECT` — see
[`docs/sn74-emission-strategy.md`](docs/sn74-emission-strategy.md)), assigned
by a deterministic bot that re-runs your scorecard on a pinned GPU, not by a
human judgment call. That bot isn't live yet — until it is, a maintainer
applies the equivalent label by hand from your reproduced scorecard. The
tiering rule is identical either way, and won't change retroactively once the
bot starts running.

---

By contributing you agree that your contribution is licensed under the project's
[MIT License](LICENSE).
