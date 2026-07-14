"""Command-line interface:  python -m strategy --n 8192 --transform rsvd ..."""
from __future__ import annotations

import argparse
import sys

from .config import Config, DTYPES
from . import runner
from .transforms import available as available_transforms, get_transform


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="strategy",
        description="Smart (subspace) matrix x matrix: compress -> compute -> "
                    "reconstruct. Approximate; for low-rank / smooth data.",
    )
    p.add_argument("--n", type=int, default=8192, help="matrix dimension n (n x n). default 8192")
    p.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--rank-m", type=int, default=None,
                   help="subspace dimension M (default min(n, max(64, n//8))). "
                        "Smaller = faster + less accurate")
    p.add_argument("--transform", default="rsvd",
                   help=f"subspace basis (the core tech). one of: "
                        f"{available_transforms()}")
    p.add_argument("--transform-seed", type=int, default=0)
    p.add_argument("--vram-fraction", type=float, default=0.6)
    p.add_argument("--compare", action="store_true",
                   help="run BOTH exact and smart on the same inputs and "
                        "report speed + accuracy")
    p.add_argument("--storage", choices=["auto", "ram", "disk"], default="auto")
    p.add_argument("--workdir", default="./_strategy_data")
    p.add_argument("--fill", choices=["lowrank", "random", "decaying-spectrum", "iota", "zeros"],
                   default="lowrank",
                   help="test-matrix content. 'lowrank' = compressible data "
                        "where the strategy is accurate (default)")
    p.add_argument("--data-rank", type=int, default=None,
                   help="rank of generated matrices for --fill lowrank / "
                        "decaying-spectrum (default max(1, n//32))")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true",
                   help="report reconstruction error vs a float64 reference")
    p.add_argument("--keep", action="store_true", help="keep on-disk files")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.n < 1:
            raise ValueError(f"--n must be a positive integer, got {args.n}")
        # Validate --transform before Config / Backend construction so an
        # unknown name exits 2 with error: … instead of a KeyError traceback
        # after GPU init (get_transform raises KeyError, which this CLI did
        # not catch — same contract as --n / --vram-fraction in #175).
        known = available_transforms()
        if args.transform not in known:
            raise ValueError(
                f"unknown transform {args.transform!r}; available: {known}"
            )
        if args.data_rank is not None and args.data_rank < 1:
            raise ValueError(f"--data-rank must be a positive integer, got {args.data_rank}")
        if args.rank_m is not None and args.rank_m < 1:
            raise ValueError(f"--rank-m must be a positive integer, got {args.rank_m}")
        cfg = Config(
            device=args.device,
            dtype=args.dtype,
            rank_m=args.rank_m,
            transform=args.transform,
            transform_seed=args.transform_seed,
            vram_fraction=args.vram_fraction,
            storage=args.storage,
            workdir=args.workdir,
            seed=args.seed,
            verbose=not args.quiet,
        )
        get_transform(cfg.transform, cfg.transform_seed)
        if args.compare:
            out = runner.compare(args.n, cfg, fill=args.fill, data_rank=args.data_rank,
                                 keep=args.keep)
            if args.quiet:
                # compare() only prints when verbose (i.e. not --quiet), so
                # without this a --compare --quiet run produced NO output at all.
                print(f"exact {out['exact_seconds']:.4f}s  "
                      f"smart {out['smart_seconds']:.4f}s  "
                      f"speedup {out['speedup']:.2f}x  rel_err {out['rel_err']:.2e}")
            return 0
        info = runner.run(args.n, cfg, fill=args.fill, verify=args.verify,
                          keep=args.keep, data_rank=args.data_rank)
    except (ValueError, RuntimeError, MemoryError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.quiet:
        err = info.get("verify", {}).get("max_rel_err")
        tail = f"  rel_err={err:.2e}" if err is not None else ""
        print(f"{info['mode']}  {info['seconds']:.4f}s{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
