# Contributing to CCO

CCO grows one way: someone submits a **strategy** that multiplies matrices for
less compute cost without losing accuracy, proves it with the shared scorer, and
opens a PR. This document is the whole loop — **the one rule**, **self-score
locally**, **submit**.

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
compression scheme, a better exact tile schedule in `matmul/`, a sharper metric
in `eval/`) are welcome too — the same one rule and the same scorecard apply.

---

## Self-score locally

Before you open a PR, run the scorer. It generates random couples, multiplies
them with the **normal (exact)** engine and your **smart** strategy on the
*identical* inputs, and prints one scorecard.

CCO computes on a **GPU** (CUDA/MPS) via PyTorch — score on a GPU machine
(reference: A100). The reference regime is **`12000`, full-rank**
(random) data, which is `eval`'s default.

```bash
# score your transform on the reference regime (12000, full-rank, 3 couples)
python -m eval --n 12000 --pairs 3 --transforms mine,rsvd

# fit the empirical time complexity O(N^p); pass --rank-m to hold M fixed (~N²),
# omit it to let M = N//8 grow with N (~N³)
python -m eval --transforms mine --rank-m 128 --sweep 512,1024,2048

# machine-readable, for pasting exact numbers
python -m eval --n 12000 --pairs 3 --transforms mine --json

# if your strategy targets compressible data, show that regime too (and say so):
python -m eval --n 12000 --pairs 3 --fill lowrank --data-rank 16 --transforms mine
```

Then confirm you did not break the gates:

```bash
python eval/tests/test_eval.py
python strategy/tests/test_subspace.py
python tests/test_correctness.py
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

### PR description format

Every PR description must be exactly this shape:

```markdown
## Summary

<what the strategy does, why it is cheaper, and the regime it targets>

## Result   (N=12000, full-rank, A100 (80 GB), fp32)

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

---

By contributing you agree that your contribution is licensed under the project's
[MIT License](LICENSE).
