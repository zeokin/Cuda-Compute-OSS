# Contributing to CCO

CCO grows one way: someone submits a **strategy** that multiplies matrices for
less compute cost without losing accuracy, proves it with the shared scorer, and
opens a PR. This document is the whole loop — **the one rule**, **self-score
locally**, **submit**.

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

Most contributions are a new **transform** — the pluggable basis that defines the
subspace the smart strategy compresses into. You add a class in
[`strategy/transforms.py`](strategy/transforms.py) and register it:

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
compression scheme, a better exact tile schedule in `matmul/`) are welcome too
— the same one rule and the same scorecard apply.

---

## What's open, what's protected

| zone | paths | policy |
|---|---|---|
| **Open — the main event** | [`strategy/transforms.py`](strategy/transforms.py) (new `Transform` classes), new strategy modules under `strategy/` | The designed hook — `register_transform()` exists for exactly this. Self-scored, verified by the scorer below. |
| **Open — engine performance** | [`matmul/`](matmul/), `strategy/backend.py`, `strategy/subspace.py` | Tiling, streaming, dtype/precision paths, platform fixes — all welcome, measured the same way. |
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

## Self-score locally

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

Before you open a PR, run the scorer. It generates random couples, multiplies
them with the **normal (exact)** engine and your **smart** strategy on the
*identical* inputs, and prints one scorecard.

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

Then confirm you did not break the gates:

```bash
uv run python eval/tests/test_eval.py
uv run python strategy/tests/test_subspace.py
uv run python tests/test_correctness.py
```

Rules for an honest local score:

- Score on **unseen** couples from the same distribution — never special-case the
  seeds, sizes, or matrices the harness uses.
- Report accuracy and latency from the **same run** at the **same dtype**.
- Use the peak-VRAM number the scorer measures; do not exclude scratch memory.
- Name the GPU (and dtype) you measured on — results depend on the device.

---

## Submit

1. **Fork & branch.** One strategy (or one focused change) per PR.
2. **Keep it standalone.** `matmul/`, `strategy/`, and `eval/` do not import each
   other except where they already do; don't add cross-coupling.
3. **Green tests.** All three test suites above must pass.
4. **Open the PR** and fill in the scorecard. The PR template
   ([`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md))
   pre-populates the exact format below — **the numbers, not the prose, decide.**
5. **Your own work.** A PR that substantially reproduces an earlier PR's diff
   is detected automatically and blocked (see
   [`.github/COPYCATS.md`](.github/COPYCATS.md)) — an independently-arrived-at
   similar solution is fine, a copy is not.

If you are coming in through Gittensor, contribute through the same GitHub path
as everyone else: code in a branch, benchmark locally, and submit a PR with
reproducible numbers.

### PR description format

Every PR description must be exactly this shape:

```markdown
## Summary

<what the strategy does, why it is cheaper, and the regime it targets>

## Result   (N=12000, full-rank, RTX 5090, fp32)

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

### Review & merge

A maintainer reproduces your scorecard on the reference setup, checks the
correctness gates and the one rule, and merges if your strategy is a genuine
improvement (or a useful strategy that documents its trade-off honestly). If the
scorecard can't be reproduced, the PR goes back for evidence — not rejected for
disagreeing with the prose.

The PR bot runs continuously for non-GPU triage. On each PR event and on a
15-minute schedule it checks drafts, blocked contributors, copycat overlap, and
scorecard presence. PRs that pass those gates get `status:queued-gpu` and appear
in `dashboard/data.json` in oldest-PR-first order. The dashboard UI itself is
expected to live in a separate private repository.

GPU evaluation is intentionally batched. The bot can run all day, but GPU tests
should run sequentially during one or two maintainer-controlled windows per day.
That keeps GPU rental predictable while preserving a public queue of what will
be tested next.

Maintainers can preview the next GPU batch without renting hardware:

```bash
uv run --extra test python -m eval.gpu_batch --limit 3
```

When a GPU is available, run the same queue sequentially:

```bash
uv run --extra test python -m eval.gpu_batch --limit 3 --run --clean
```

By default the GPU scorer omits `--seed`, so every official run sees fresh
unseen matrices. Pass `--seed <n>` only to reproduce a prior result.

**Scoring is moving to automated verdict labels** (`eval:XS` through `eval:XL`,
plus `eval:BASELINE`, `eval:none`, `eval:REJECT` — see
[`docs/sn74-emission-strategy.md`](docs/sn74-emission-strategy.md)), assigned
by a deterministic bot that re-runs your scorecard on a pinned GPU, not by a
human judgment call. That bot isn't live yet — until it is, a maintainer
applies the equivalent label by hand from your reproduced scorecard. The
tiering rule is identical either way, and won't change retroactively once the
bot starts running.

---

By contributing you agree that your contribution is licensed under the project's
[MIT License](LICENSE).
