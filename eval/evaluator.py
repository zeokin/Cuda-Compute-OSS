"""Evaluation harness for the smart (subspace) strategy.

Pipeline (one pass):

    1. generate ``pairs`` random couples (A_i, B_i), each (N, N)
    2. C_i  = A_i @ B_i          via NORMAL computing (exact, streamed)
    3. Ĉ_i  = subspace(A_i, B_i) via SMART computing (per transform)
    4. estimate each transform: accuracy, latency, peak VRAM, FLOP complexity
    5. score = accuracy × (1/Peak_VRAM) × (1/Latency), gated by an accuracy floor

The exact products are computed once and reused across every transform so the
comparison is apples-to-apples on identical inputs.

Standalone-ish: imports only the sibling ``strategy`` package (the thing under
test) plus this folder's metric/memory helpers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from strategy import transforms as _transforms
from strategy.backend import Backend
from strategy.config import Config
from strategy import subspace, storage

from . import metrics
from .memory import MemoryProbe


@dataclass
class EvalConfig:
    """Knobs for one evaluation sweep.

    n            : matrix edge N (square N x N couples).
    pairs        : how many random couples to generate and average over.
    dtype        : element type (fp16 / fp32 / fp64).
    rank_m       : subspace dimension M for the smart strategy (None => N//8).
    fill         : matrix content: 'random' (hard, full-rank), 'lowrank'
                   (the strategy's happy path), or 'iota'.
    data_rank    : rank used when fill='lowrank' (None => N//32).
    transforms   : transform names to evaluate (None => all registered).
    accuracy_floor: accuracy below this hard-gates the score to 0 (default 0.8;
                    set 0.0 to disable the gate).
    vram_unit    : unit for Peak_VRAM inside the score ('gib'|'mib'|'bytes').
    seed         : base RNG seed (pair i uses seed+2i, seed+2i+1).
    device       : GPU device index (CUDA). Compute runs on GPU via PyTorch.
    verbose      : print the report.
    """

    n: int = 12000
    pairs: int = 3
    dtype: str = "fp32"
    rank_m: int | None = None
    fill: str = "random"          # full-rank by default (the honest, general case)
    data_rank: int | None = None
    transforms: list[str] | None = None
    accuracy_floor: float = 0.8
    vram_unit: str = "gib"
    seed: int = 0
    device: int = 0
    verbose: bool = True


def _strategy_config(ev: EvalConfig, transform: str) -> Config:
    return Config(
        device=ev.device,
        dtype=ev.dtype,
        rank_m=ev.rank_m,
        transform=transform,
        vram_fraction=0.6,
        storage="ram",
        seed=ev.seed,
        verbose=False,
    )


def _timed_with_mem(fn, backend):
    """Run ``fn`` returning (seconds, peak_bytes, fn_result)."""
    import time

    probe = MemoryProbe(backend)
    backend.synchronize()
    with probe:
        t0 = time.perf_counter()
        result = fn()
        backend.synchronize()
        seconds = time.perf_counter() - t0
    return seconds, probe.peak_bytes, result


def _generate_pairs(ev: EvalConfig):
    """Return a list of (A, B) NumPy couples, each (n, n), in RAM."""
    dt = np.dtype({"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}[ev.dtype])
    pairs = []
    for i in range(ev.pairs):
        A = storage.generate(ev.n, dt, False, None, ev.seed + 2 * i, ev.fill,
                             data_rank=ev.data_rank)
        B = storage.generate(ev.n, dt, False, None, ev.seed + 2 * i + 1, ev.fill,
                             data_rank=ev.data_rank)
        pairs.append((A, B))
    return pairs, dt


def evaluate(ev: EvalConfig) -> dict:
    """Run the full evaluation and return a results dict (see module docstring)."""
    backend = Backend(ev.device, ev.verbose)
    names = ev.transforms or _transforms.available()

    if ev.verbose:
        print(f"[eval] device     : {backend.name}")
        print(f"[eval] couples     : {ev.pairs} x ({ev.n} x {ev.n})  fill={ev.fill}  "
              f"dtype={ev.dtype}")
        print(f"[eval] rank_m (M)  : {ev.rank_m or ev.n // 8}")
        print(f"[eval] transforms  : {', '.join(names)}")

    pairs, dt = _generate_pairs(ev)

    # ---- normal computing: exact products, computed once, reused ----------
    cfg0 = _strategy_config(ev, names[0])
    exact_products, exact_lat, exact_vram = [], [], []
    for (A, B) in pairs:
        Ce = np.empty((ev.n, ev.n), dtype=dt)
        sec, peak, _ = _timed_with_mem(
            lambda A=A, B=B, Ce=Ce: subspace.multiply_exact(A, B, Ce, backend, cfg0),
            backend,
        )
        exact_products.append(Ce)
        exact_lat.append(sec)
        exact_vram.append(peak)

    # exact baseline the dominance gate compares every strategy against: mean
    # latency and worst-case VRAM, measured on the identical couples.
    exact_latency = float(np.mean(exact_lat))
    exact_peak_vram = float(np.max(exact_vram))

    # ---- smart computing: one entry per transform -------------------------
    results = {}
    for name in names:
        cfg = _strategy_config(ev, name)
        accs, errs, lats, vrams, flop_ratio = [], [], [], [], None
        for (A, B), Ce in zip(pairs, exact_products):
            Cs = np.empty((ev.n, ev.n), dtype=dt)
            sec, peak, info = _timed_with_mem(
                lambda A=A, B=B, Cs=Cs: subspace.multiply_subspace(A, B, Cs, backend, cfg),
                backend,
            )
            err = metrics.rel_frobenius_error(Ce, Cs)
            errs.append(err)
            accs.append(max(0.0, 1.0 - err))
            lats.append(sec)
            vrams.append(peak)
            flop_ratio = info["flop_exact"] / info["flop_actual"]

        acc = float(np.mean(accs))
        rel_err = float(np.mean(errs))
        latency = float(np.mean(lats))
        peak_vram = float(np.max(vrams))          # worst-case memory pick

        # The improvement rule (BENCHMARKS.md): a strategy is admitted only if
        # accuracy clears the floor AND it dominates the exact baseline on every
        # cost axis. The raw performance figure is kept for transparency, but the
        # ranking ``score`` is zero for anything that is not an improvement — an
        # accurate-but-slower or heavier strategy does not beat exact.
        gated = acc < ev.accuracy_floor
        cost_dominant = metrics.dominates_exact(
            latency, peak_vram, flop_ratio, exact_latency, exact_peak_vram
        )
        improvement = (not gated) and cost_dominant
        perf_score = metrics.score(
            acc, peak_vram, latency, ev.accuracy_floor, ev.vram_unit
        )
        results[name] = {
            "accuracy": acc,
            "rel_frobenius_error": rel_err,
            "latency_s": latency,
            "peak_vram_bytes": peak_vram,
            "peak_vram_mib": peak_vram / metrics.MIB,
            "flop_ratio_vs_exact": flop_ratio,
            "faster_than_exact": latency < exact_latency,
            "less_vram_than_exact": peak_vram < exact_peak_vram,
            "fewer_flops_than_exact": (flop_ratio or 0.0) > 1.0,
            "gated": gated,
            "improvement": improvement,
            "perf_score": perf_score,
            "score": perf_score if improvement else 0.0,
        }

    ranking = sorted(results.items(), key=lambda kv: kv[1]["score"], reverse=True)
    m = ev.rank_m or ev.n // 8
    out = {
        "config": {
            "n": ev.n, "pairs": ev.pairs, "dtype": ev.dtype, "rank_m": m,
            "fill": ev.fill, "accuracy_floor": ev.accuracy_floor,
            "vram_unit": ev.vram_unit, "device": backend.name,
        },
        "complexity": {
            "normal": "O(N^3)",
            "smart": "O(N^2 * M)"
            + (f"  (M={m} fixed -> ~O(N^2))" if ev.rank_m else f"  (M=N/8 -> ~O(N^3))"),
        },
        "exact": {
            "latency_s": float(np.mean(exact_lat)),
            "peak_vram_bytes": float(np.max(exact_vram)),
            "peak_vram_mib": float(np.max(exact_vram)) / metrics.MIB,
        },
        "transforms": results,
        "ranking": [name for name, _ in ranking],
        "best": ranking[0][0] if ranking else None,
    }
    if ev.verbose:
        _print_report(out)
    return out


def estimate_scaling(ns, ev: EvalConfig) -> dict:
    """Empirically fit the smart strategy's time complexity O(N^p).

    Runs one smart multiply per N in ``ns`` and fits latency ~ c * N^p by least
    squares in log-log space. Set ``ev.rank_m`` to hold M fixed (isolates the
    ~N^2 term); leave it None and M = N//8 grows with N (~N^3).
    """
    backend = Backend(ev.device, ev.verbose)
    name = (ev.transforms or _transforms.available())[0]
    lat_by_n = {}
    for n in ns:
        sub = EvalConfig(**{**ev.__dict__, "n": n, "pairs": 1, "verbose": False})
        cfg = _strategy_config(sub, name)
        A, B = _generate_pairs(sub)[0][0]
        Cs = np.empty((n, n), dtype=A.dtype)
        sec, _, _ = _timed_with_mem(
            lambda: subspace.multiply_subspace(A, B, Cs, backend, cfg), backend
        )
        lat_by_n[n] = sec

    xs = np.log(np.array(list(lat_by_n.keys()), dtype=np.float64))
    ys = np.log(np.array(list(lat_by_n.values()), dtype=np.float64))
    p, _c = np.polyfit(xs, ys, 1) if len(xs) >= 2 else (float("nan"), 0.0)
    out = {"transform": name, "latency_by_n": lat_by_n,
           "fitted_exponent_p": float(p), "model": "latency ~ N^p"}
    if ev.verbose:
        print(f"\n[eval] scaling of '{name}': latency ~ N^{p:.2f}")
        for n, s in lat_by_n.items():
            print(f"         N={n:<7} {s*1e3:8.2f} ms")
    return out


# ---------------------------------------------------------------------------
def _print_report(out: dict) -> None:
    c = out["config"]
    print(f"\n=== eval report (n={c['n']}, {c['pairs']} couples, {c['dtype']}, "
          f"fill={c['fill']}, M={c['rank_m']}) ===")
    print(f"  complexity : normal {out['complexity']['normal']}   "
          f"smart {out['complexity']['smart']}")
    print(f"  exact      : {out['exact']['latency_s']*1e3:8.2f} ms   "
          f"peak {out['exact']['peak_vram_mib']:8.1f} MiB")
    header = f"  {'transform':<10}{'accuracy':>10}{'latency':>13}" \
             f"{'peakVRAM':>11}{'FLOPx':>8}{'score':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in out["ranking"]:
        r = out["transforms"][name]
        if r["gated"]:
            note = "  (gated: low accuracy)"
        elif not r["improvement"]:
            regressed = [ax for ax, ok in (
                ("latency", r["faster_than_exact"]),
                ("VRAM", r["less_vram_than_exact"]),
                ("FLOPs", r["fewer_flops_than_exact"]),
            ) if not ok]
            note = f"  (not an improvement: {', '.join(regressed)} ≥ exact)"
        else:
            note = "  (improvement)"
        print(f"  {name:<10}{r['accuracy']:>10.4f}"
              f"{r['latency_s']*1e3:>11.2f}ms{r['peak_vram_mib']:>9.1f}MiB"
              f"{r['flop_ratio_vs_exact']:>7.1f}x{r['score']:>13.4g}{note}")
    print(f"  best (highest score): {out['best']}")
