"""
cco/significance.py — challenger-vs-champion win decision (Step 6).

A challenger kernel takes the crown only if it is BOTH:
  1. statistically faster than the current champion (low p-value), AND
  2. faster by at least a real margin (effect size above the GPU noise floor).

The test is a **one-sided Mann-Whitney U** on the two latency samples (lower latency = faster),
deliberately NOT a Welch t-test: GPU clock-boost is often bimodal, which violates the t-test's
normality assumption and makes it misfire (a red-team finding). Mann-Whitney is nonparametric —
it only assumes the samples are comparable — so bimodal boost doesn't break it. The samples are
the per-block median latencies that benchmark.py emits (champion and challenger re-run fresh and
interleaved in the same sealed job, so they share thermal state).

This decision is owned by the maintainer agent (it consumes two score blobs); benchmark.py only
produces samples and never decides. Pure stdlib (math/statistics) — no scipy/numpy dependency.

Usage:
    uv run --no-project python cco/significance.py --self-test
"""

from __future__ import annotations

import math
import statistics
from collections import Counter

DEFAULT_MIN_IMPROVEMENT_PCT = 5.0
DEFAULT_P_THRESHOLD = 0.01


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _average_ranks(vals: list[float]) -> list[float]:
    """1-based ranks, averaged within tie groups, aligned to input order."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i, n = 0, len(vals)
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = ((i + 1) + (j + 1)) / 2.0  # average of 1-based ranks across the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def mannwhitney_p_less(a: list[float], b: list[float]) -> float:
    """One-sided p-value that sample `a` is stochastically LESS than `b` (a tends lower).

    Normal approximation with tie correction (good for n >= ~20; we use n=30 blocks).
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 1.0
    combined = a + b
    ranks = _average_ranks(combined)
    r1 = sum(ranks[:n1])                       # rank sum of `a`
    u1 = r1 - n1 * (n1 + 1) / 2.0              # small when `a` has low ranks (low values)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie_term = sum(t ** 3 - t for t in Counter(combined).values())
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return 0.5
    z = (u1 - mu) / math.sqrt(var)
    return _normal_cdf(z)                       # P(a stochastically <= b)


def challenger_wins(
    champion_latencies: list[float],
    challenger_latencies: list[float],
    min_improvement_pct: float = DEFAULT_MIN_IMPROVEMENT_PCT,
    p_threshold: float = DEFAULT_P_THRESHOLD,
) -> dict:
    """Decide whether the challenger beats the champion. Latencies: lower = faster."""
    med_c = statistics.median(champion_latencies)
    med_x = statistics.median(challenger_latencies)
    speedup = (med_c / med_x) if med_x > 0 else float("inf")
    improvement_pct = (speedup - 1.0) * 100.0
    p = mannwhitney_p_less(challenger_latencies, champion_latencies)  # challenger faster?
    significant = p < p_threshold
    margin_met = improvement_pct >= min_improvement_pct
    return {
        "win": bool(significant and margin_met),
        "significant": significant,
        "margin_met": margin_met,
        "p_value": p,
        "speedup": speedup,
        "improvement_pct": improvement_pct,
        "median_champion": med_c,
        "median_challenger": med_x,
        "n_champion": len(champion_latencies),
        "n_challenger": len(challenger_latencies),
    }


def load_thresholds_from_config(config_path: str) -> tuple[float, float]:
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        sig = json.load(f)["scoring"]["significance"]
    return float(sig["min_improvement_pct"]), float(sig["p_value_threshold"])


# --------------------------------------------------------------------------------------
# Self-test (pure Python; seeded synthetic latency samples)
# --------------------------------------------------------------------------------------

def _self_test() -> int:
    import random
    random.seed(0)

    def sample(mean, sd, n=30):
        return [random.gauss(mean, sd) for _ in range(n)]

    # (label, champ, challenger, expect_win, extra-assertions)
    cases = [
        ("identical distributions",         sample(100, 2),  sample(100, 2),  False),
        ("challenger 10% faster (clean)",   sample(100, 2),  sample(90, 2),   True),
        ("challenger ~1% faster (< margin)",sample(100, 0.5),sample(99, 0.5), False),  # significant but margin fails
        ("challenger slower",               sample(100, 2),  sample(112, 2),  False),
        ("challenger 2% faster, very noisy",sample(100, 12), sample(98, 12),  False),  # not significant
    ]

    failures = 0
    for label, champ, chal, expect in cases:
        r = challenger_wins(champ, chal)
        ok = (r["win"] == expect)
        flag = "ok  " if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"{flag} {label:36s} win={r['win']!s:5s} "
              f"(p={r['p_value']:.4f} improvement={r['improvement_pct']:+.1f}% "
              f"sig={r['significant']} margin={r['margin_met']})")

    print("-" * 70)
    print("SELF-TEST PASSED" if not failures else f"SELF-TEST FAILED: {failures} case(s)")
    return 1 if failures else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Challenger-vs-champion significance decision (CCO).")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    p.error("pass --self-test (or import challenger_wins / mannwhitney_p_less)")


if __name__ == "__main__":
    raise SystemExit(main())
