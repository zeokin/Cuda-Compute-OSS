"""High-level run orchestration: allocate A/B/C, multiply, optionally verify
and benchmark. Used by the CLI and by examples."""
from __future__ import annotations

import os
import time
import numpy as np

from .backend import Backend
from .config import Config
from . import gemm, storage


def _paths(workdir: str):
    return (
        os.path.join(workdir, "A.dat"),
        os.path.join(workdir, "B.dat"),
        os.path.join(workdir, "C.dat"),
    )


def run(n: int, cfg: Config, fill: str = "random",
        verify: bool = False, keep: bool = False) -> dict:
    """Generate two n x n matrices, compute C = A @ B, and report.

    Returns a dict with timing, GFLOP/s, and (if verify) accuracy vs a float64
    reference. verify only runs for small n (the reference is computed on CPU).
    """
    backend = Backend(cfg.device, cfg.verbose)
    dt = cfg.np_dtype
    on_disk = storage.should_use_disk(
        n, cfg.item_bytes, cfg.storage, backend.host_available_bytes()
    )
    pa, pb, pc = _paths(cfg.workdir)

    if cfg.verbose:
        print(f"[run] device      : {backend.name}")
        print(f"[run] n            : {n}  ({cfg.dtype})")
        print(f"[run] per-matrix   : {storage.bytes_human(n * n * cfg.item_bytes)}"
              f"  (A+B+C = {storage.bytes_human(3 * n * n * cfg.item_bytes)})")
        print(f"[run] storage      : {'disk memmap' if on_disk else 'RAM'}"
              f"{' @ ' + cfg.workdir if on_disk else ''}")

    A = storage.generate(n, dt, on_disk, pa if on_disk else None, cfg.seed, fill)
    B = storage.generate(n, dt, on_disk, pb if on_disk else None, cfg.seed + 1, fill)
    C = storage.allocate(n, dt, on_disk, pc if on_disk else None)

    backend.synchronize()
    t0 = time.perf_counter()
    info = gemm.multiply(A, B, C, backend, cfg)
    backend.synchronize()
    elapsed = time.perf_counter() - t0

    flop = 2.0 * n * n * n
    info.update(
        seconds=elapsed,
        gflops=flop / elapsed / 1e9,
        storage="disk" if on_disk else "ram",
    )

    if cfg.verbose:
        print(f"[run] mode         : {info['mode']}")
        print(f"[run] time         : {elapsed:.4f} s")
        print(f"[run] throughput   : {info['gflops']:.1f} GFLOP/s")

    if verify:
        info["verify"] = _verify(A, B, C, n, cfg)
        if cfg.verbose:
            v = info["verify"]
            print(f"[run] verify       : max_rel_err={v['max_rel_err']:.2e} "
                  f"({'OK' if v['ok'] else 'MISMATCH'})")

    if on_disk and not keep:
        for p in (pa, pb, pc):
            try:
                os.remove(p)
            except OSError:
                pass

    return info


def _verify(A, B, C, n: int, cfg: Config) -> dict:
    """Compare against a float64 CPU reference. Only sensible for small n."""
    ref = np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)
    got = np.asarray(C, dtype=np.float64)
    denom = np.linalg.norm(ref) or 1.0
    rel_err = float(np.linalg.norm(got - ref) / denom)
    # fp16 carries large rounding error; scale tolerance by dtype.
    tol = {"fp16": 5e-2, "fp32": 1e-4, "fp64": 1e-10}[cfg.dtype]
    return {"max_rel_err": rel_err, "tol": tol, "ok": rel_err <= tol}
