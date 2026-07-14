<!--
CCO has two PR lanes:
  - fix/bug PRs: CPU-safe validation only, no GPU scorecard required
  - feat/strategy PRs: score-bearing, GPU scorecard required

Read CONTRIBUTING.md and BENCHMARKS.md before filling this in.
The one rule: an improvement reduces every cost axis WITHOUT losing accuracy.
-->

## PR kind

- [ ] fix
- [ ] feat

<!-- Check exactly one box. The bot and CI use this to route the PR. -->

## Summary

<!--
For fix PRs: explain the bug and the behavior you corrected.
For feat PRs: explain what the strategy/feature does, why it is cheaper, and
the regime it targets.
-->

## Validation

<!--
List the CPU-safe validation you ran.
Example:
- uv run python -m strategy.smoke
- uv run --extra test python -m pytest tests/ strategy/tests/ eval/tests/ -v
-->

## Target track (feat PRs — declare exactly one)

- [ ] full-rank — random data, the hard general case (accuracy floor 0.80)
- [ ] low-rank — rank ≪ N (accuracy floor 0.95)
- [ ] decaying-spectrum — polynomially decaying singular values (accuracy floor 0.90)

<!--
The GPU bot re-scores your PR at this track's PINNED regime (fixed rank / M /
data), on FRESH unseen seeds, rebased onto current `main`, and computes the tier
against the ledger's recorded frontier. So the numbers you paste below are
context, not the verdict — you cannot pick the rank/M that flatters your method,
and the baseline is always the current frontier, never the rsvd on your branch.
-->

**Transform:** `____`

<!--
The transform your PR adds or changes (the name in `register_transform("…")`),
e.g. `nystrom`. The bot scores THIS transform, and verifies your diff actually
adds or modifies it — you cannot claim credit for a transform you did not write.
-->

## GPU Result (required for feat PRs only)

| metric          | value          |
|-----------------|----------------|
| accuracy        |                |
| time complexity |                |
| latency         |                |
| VRAM usage      |                |

<!--
accuracy        — bounded Frobenius accuracy in [0,1] from `python -m eval`
time complexity — analytic O(·) and the fitted N^p from `--sweep`
latency         — mean wall-clock ms of the smart multiply, GPU-synchronized
VRAM usage      — peak incremental GPU memory during the multiply
-->

**Regime measured:** N=8192, dtype=fp32, fill=full-rank, rank M=____, device=RTX 5090

<details>
<summary>Raw scorecard (paste <code>python -m eval …</code> output or <code>--json</code>)</summary>

```
<paste here>
```
</details>

## Checklist

- [ ] CPU-safe validation passed (`strategy.smoke` if relevant, plus `pytest tests/ strategy/tests/ eval/tests/`).
- [ ] My commits do not include `Co-authored-by` footers for coding agents such as Cursor, Codex, Claude, Copilot, or similar tools.
- [ ] If this is a feat PR, I ran the scorer on **unseen** couples — no hardcoding of seeds/matrices.
- [ ] If this is a feat PR, accuracy and latency come from the **same run** at the **same dtype**.
- [ ] If this is a feat PR, this is an **improvement** (every cost axis down, accuracy held) **or** I
      state honestly which axis it trades — see the one rule in CONTRIBUTING.md.
- [ ] Correctness gates pass:
      `python eval/tests/test_eval.py`,
      `python strategy/tests/test_subspace.py`,
      `python tests/test_correctness.py`.
- [ ] If this is a feat PR, I named the device and dtype so a reviewer can reproduce the numbers.
