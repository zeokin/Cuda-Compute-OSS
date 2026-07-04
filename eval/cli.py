"""Command-line entry point:  python -m eval

Evaluate the smart (subspace) strategy's transforms on random matrix couples.
"""
from __future__ import annotations

import argparse
import json
import sys

from strategy import transforms as _transforms

from .evaluator import EvalConfig, evaluate, estimate_scaling


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval",
        description="Score the smart (subspace) strategy vs normal (exact) "
                    "computing: accuracy, latency, peak VRAM -> composite score.",
    )
    p.add_argument("--n", type=int, default=12000, help="matrix edge N (N x N). default 12000")
    p.add_argument("--pairs", type=int, default=3,
                   help="number of random couples to average over. default 3")
    p.add_argument("--dtype", choices=("fp16", "fp32", "fp64"), default="fp32")
    p.add_argument("--rank-m", type=int, default=None,
                   help="subspace dimension M for the smart strategy "
                        "(default min(N, max(64, N//8)))")
    p.add_argument("--fill", choices=("random", "lowrank", "iota"), default="random",
                   help="matrix content. 'random'=hard/full-rank (default), "
                        "'lowrank'=the strategy's happy path")
    p.add_argument("--data-rank", type=int, default=None,
                   help="rank used when --fill lowrank (default N//32)")
    p.add_argument("--transforms", default=None,
                   help="comma-separated transform names (default: all registered: "
                        + ",".join(_transforms.available()) + ")")
    p.add_argument("--min-accuracy", type=float, default=0.8,
                   help="accuracy below this hard-gates a transform's score to 0 "
                        "(default 0.8; use 0 to disable)")
    p.add_argument("--vram-unit", choices=("bytes", "mib", "gib"), default="gib",
                   help="unit for Peak_VRAM inside the score. default gib")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--sweep", default=None,
                   help="comma-separated N list to fit empirical O(N^p), e.g. 128,256,512")
    p.add_argument("--json", action="store_true", help="print results as JSON")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(argv)
    ev = EvalConfig(
        n=a.n, pairs=a.pairs, dtype=a.dtype, rank_m=a.rank_m, fill=a.fill,
        data_rank=a.data_rank,
        transforms=[s.strip() for s in a.transforms.split(",")] if a.transforms else None,
        accuracy_floor=a.min_accuracy, vram_unit=a.vram_unit, seed=a.seed,
        device=a.device, verbose=not a.json,
    )
    try:
        out = evaluate(ev)
        if a.sweep:
            ns = [int(x) for x in a.sweep.split(",")]
            out["scaling"] = estimate_scaling(ns, ev)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
