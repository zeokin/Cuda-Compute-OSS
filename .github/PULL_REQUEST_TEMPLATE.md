<!--
CCO PR scorecard. The numbers, not the prose, decide.
Read CONTRIBUTING.md and BENCHMARKS.md before filling this in.
The one rule: an improvement reduces every cost axis WITHOUT losing accuracy.
-->

## Summary

<!-- What the strategy does, why it is cheaper, and the regime it targets. -->

## Result

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

**Regime measured:** N=12000, dtype=fp32, fill=full-rank, rank M=____, device=A100 (80 GB)

<details>
<summary>Raw scorecard (paste <code>python -m eval …</code> output or <code>--json</code>)</summary>

```
<paste here>
```
</details>

## Checklist

- [ ] I ran the scorer on **unseen** couples — no hardcoding of seeds/matrices.
- [ ] Accuracy and latency come from the **same run** at the **same dtype**.
- [ ] This is an **improvement** (every cost axis down, accuracy held) **or** I
      state honestly which axis it trades — see the one rule in CONTRIBUTING.md.
- [ ] Correctness gates pass:
      `python eval/tests/test_eval.py`,
      `python strategy/tests/test_subspace.py`,
      `python tests/test_correctness.py`.
- [ ] I named the device and dtype so a reviewer can reproduce the numbers.
