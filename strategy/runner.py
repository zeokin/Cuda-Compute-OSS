"""Run orchestration for the subspace strategy: allocate A/B/C, run the smart
multiply (and optionally the exact baseline), verify and report.

Standalone: no imports from the sibling `matmul` package.
"""
from __future__ import annotations

import os
import time
import numpy as np

from .backend import Backend
from .config import Config
from . import subspace, storage


def _paths(workdir: str):
    return (
        os.path.join(workdir, "A.dat"),
        os.path.join(workdir, "B.dat"),
        os.path.join(workdir, "C.dat"),
    )


def _timed(fn, backend) -> float:
    backend.synchronize()
    t0 = time.perf_counter()
    fn()
    backend.synchronize()
    return time.perf_counter() - t0


def run(n: int, cfg: Config, fill: str = "lowrank", verify: bool = False,
        keep: bool = False, data_rank: int | None = None) -> dict:
    """Generate A, B, compute C = A @ B with the subspace strategy, report.

    Default fill is 'lowrank' because that is the regime the strategy targets;
    verify (small n) reports the reconstruction error vs a float64 reference.
    """
    backend = Backend(cfg.device, cfg.verbose)
    dt = cfg.np_dtype
    on_disk = storage.should_use_disk(
        n, cfg.item_bytes, cfg.storage, backend.host_available_bytes()
    )
    pa, pb, pc = _paths(cfg.workdir)

    if cfg.verbose:
        print(f"[strategy] device    : {backend.name}")
        print(f"[strategy] n          : {n}  ({cfg.dtype}), fill={fill}")
        print(f"[strategy] per-matrix : {storage.bytes_human(n * n * cfg.item_bytes)}"
              f"  (A+B+C = {storage.bytes_human(3 * n * n * cfg.item_bytes)})")
        print(f"[strategy] storage    : {'disk memmap' if on_disk else 'RAM'}"
              f"{' @ ' + cfg.workdir if on_disk else ''}")

    A = storage.generate(n, dt, on_disk, pa if on_disk else None, cfg.seed, fill,
                         data_rank=data_rank)
    B = storage.generate(n, dt, on_disk, pb if on_disk else None, cfg.seed + 1, fill,
                         data_rank=data_rank)
    C = storage.allocate(n, dt, on_disk, pc if on_disk else None)

    info = {}
    elapsed = _timed(lambda: info.update(subspace.multiply_subspace(A, B, C, backend, cfg)),
                     backend)

    info.update(seconds=elapsed, gflops=2.0 * n**3 / elapsed / 1e9,
                storage="disk" if on_disk else "ram")

    if cfg.verbose:
        print(f"[strategy] mode       : {info['mode']}")
        print(f"[strategy] time       : {elapsed:.4f} s")
        print(f"[strategy] equiv-tput : {info['gflops']:.1f} GFLOP/s (vs 2N^3 work)")
        print(f"[strategy] flop saved : {info['flop_exact'] / info['flop_actual']:.1f}x "
              f"fewer FLOPs than exact")

    if verify:
        info["verify"] = _verify(A, B, C, n, cfg, backend)
        if cfg.verbose:
            v = info["verify"]
            if v.get("skipped"):
                print(f"[strategy] rel. error : skipped ({v['skipped']})")
            else:
                print(f"[strategy] rel. error : {v['max_rel_err']:.2e} "
                      f"(approximate by design)")

    if on_disk and not keep:
        for p in (pa, pb, pc):
            try:
                os.remove(p)
            except OSError:
                pass
    return info


def _rel_frobenius_streamed(Ce, Cs, block_bytes: int = 256 * 1024**2) -> float:
    """Relative Frobenius error ||Cs - Ce||_F / ||Ce||_F, one row-block at a time.

    ``Ce``/``Cs`` may be disk-backed memmaps far larger than RAM (that is the whole
    point of the streaming engine), so we must never cast the full (n, n) product to
    float64 at once -- doing so force-loads both matrices plus a diff temporary into
    host RAM (~3*n^2*8 bytes) and OOMs the host. The Frobenius norm is separable over
    rows, so accumulating squared sums block-by-block is numerically identical while
    keeping only one float64 row-block of each operand resident."""
    n = Ce.shape[0]
    row_bytes = max(1, n * 8)                       # one float64 row
    blk = max(1, min(n, block_bytes // row_bytes))
    num_sq = 0.0
    den_sq = 0.0
    for r0 in range(0, n, blk):
        r1 = min(n, r0 + blk)
        ce = np.asarray(Ce[r0:r1], dtype=np.float64)
        cs = np.asarray(Cs[r0:r1], dtype=np.float64)
        num_sq += float(np.sum((cs - ce) ** 2))
        den_sq += float(np.sum(ce ** 2))
    den = np.sqrt(den_sq)
    return float(np.sqrt(num_sq) / (den or 1.0))


def compare(n: int, cfg: Config, fill: str = "lowrank",
            data_rank: int | None = None, keep: bool = False) -> dict:
    """Run the exact baseline and the subspace strategy on the SAME inputs and
    report time, throughput and the smart strategy's reconstruction error."""
    backend = Backend(cfg.device, cfg.verbose)
    dt = cfg.np_dtype
    on_disk = storage.should_use_disk(
        n, cfg.item_bytes, cfg.storage, backend.host_available_bytes()
    )
    pa, pb, pe, ps = (*_paths(cfg.workdir)[:2],
                      os.path.join(cfg.workdir, "Ce.dat"),
                      os.path.join(cfg.workdir, "Cs.dat"))

    A = storage.generate(n, dt, on_disk, pa if on_disk else None, cfg.seed, fill,
                         data_rank=data_rank)
    B = storage.generate(n, dt, on_disk, pb if on_disk else None, cfg.seed + 1, fill,
                         data_rank=data_rank)
    Ce = storage.allocate(n, dt, on_disk, pe if on_disk else None)
    Cs = storage.allocate(n, dt, on_disk, ps if on_disk else None)

    ex = {}
    ex_t = _timed(lambda: ex.update(subspace.multiply_exact(A, B, Ce, backend, cfg)), backend)
    sm = {}
    sm_t = _timed(lambda: sm.update(subspace.multiply_subspace(A, B, Cs, backend, cfg)), backend)

    rel_err = _rel_frobenius_streamed(Ce, Cs)
    speedup = ex_t / sm_t if sm_t else float("inf")
    out = {
        "n": n, "exact_seconds": ex_t, "smart_seconds": sm_t,
        "speedup": speedup, "rel_err": rel_err,
        "flop_ratio": sm["flop_exact"] / sm["flop_actual"],
        "exact_mode": ex["mode"], "smart_mode": sm["mode"],
    }
    if cfg.verbose:
        print("\n--- normal (exact) vs smart (subspace) ---")
        print(f"  exact : {ex_t:.4f}s   ({ex['mode']})")
        print(f"  smart : {sm_t:.4f}s   ({sm['mode']})")
        print(f"  speedup(exact/smart) : {speedup:.2f}x")
        print(f"  smart FLOPs          : {out['flop_ratio']:.1f}x fewer than exact")
        print(f"  smart rel. error     : {rel_err:.2e}")

    if on_disk and not keep:
        for p in (pa, pb, pe, ps):
            try:
                os.remove(p)
            except OSError:
                pass
    return out


def _verify(A, B, C, n: int, cfg: Config, backend: Backend) -> dict:
    """Reconstruction error vs a float64 CPU reference. Small n only:
    it materializes A, B, the reference product and C as float64 in host RAM
    (~4*n*n*8 bytes) and runs an O(n^3) CPU multiply, so it is SKIPPED when that
    working set would not fit safely -- otherwise --verify on a large / disk-backed
    run OOMs the host after the smart multiply already succeeded."""
    need = 4 * n * n * 8                      # A_f64, B_f64, ref, got
    host_free = backend.host_available_bytes()
    if need > 0.5 * host_free:
        return {"skipped": f"n={n}: float64 CPU reference needs ~{storage.bytes_human(need)}, "
                f"over 50% of ~{storage.bytes_human(host_free)} host RAM"}
    ref = np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)
    got = np.asarray(C, dtype=np.float64)
    rel_err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) or 1.0))
    return {"max_rel_err": rel_err}
