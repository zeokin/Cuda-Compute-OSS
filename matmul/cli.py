"""Command-line interface:  python -m matmul --n 128000 --dtype fp32 ..."""
from __future__ import annotations

import argparse
import sys

from .config import Config, DTYPES
from . import runner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matmul",
        description="GPU matrix x matrix (square n x n) with out-of-core tiling.",
    )
    p.add_argument("--n", type=int, default=8192, help="matrix dimension n (n x n). default 8192")
    p.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--tile", type=int, default=None,
                   help="tile edge T (default: auto from free VRAM)")
    p.add_argument("--vram-fraction", type=float, default=0.6,
                   help="fraction of free VRAM tiles may use (default 0.6)")
    p.add_argument("--force-tiled", action="store_true",
                   help="always use the tiled path even if it fits in-core")
    p.add_argument("--storage", choices=["auto", "ram", "disk"], default="auto")
    p.add_argument("--workdir", default="./_matmul_data",
                   help="directory for on-disk memmap files")
    p.add_argument("--fill", choices=["random", "iota", "zeros"], default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true",
                   help="check result vs a float64 reference (small n only)")
    p.add_argument("--keep", action="store_true", help="keep on-disk files after run")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.n < 1:
            raise ValueError(f"--n must be a positive integer, got {args.n}")
        cfg = Config(
            device=args.device,
            dtype=args.dtype,
            tile=args.tile,
            vram_fraction=args.vram_fraction,
            force_tiled=args.force_tiled,
            storage=args.storage,
            workdir=args.workdir,
            seed=args.seed,
            verbose=not args.quiet,
        )
        info = runner.run(args.n, cfg, fill=args.fill, verify=args.verify, keep=args.keep)
    except (ValueError, RuntimeError, MemoryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.quiet:
        print(f"{info['mode']}  {info['seconds']:.4f}s  {info['gflops']:.1f} GFLOP/s")
    # Fail only on an explicit mismatch. A skipped verify (large / disk-backed n,
    # where the float64 CPU reference won't fit) returns {"skipped": ...} with no
    # "ok" key, and a pass returns ok=True -- neither is a failure.
    v = info.get("verify")
    if v and v.get("ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
