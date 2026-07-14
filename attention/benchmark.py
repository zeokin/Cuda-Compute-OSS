"""Benchmark exact attention against the hybrid reference operator."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

from .data import generate_qkv, resolve_device
from .spec import AttentionSpec


def _torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The attention playground requires PyTorch. Install the GPU extra: "
            "uv sync --extra gpu"
        ) from exc
    return torch


def _synchronize(dev) -> None:
    torch = _torch()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def _peak_bytes(dev) -> int:
    torch = _torch()
    if dev.type == "cuda":
        return int(torch.cuda.max_memory_allocated(dev))
    return 0


def _timed(fn, dev):
    torch = _torch()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    _synchronize(dev)
    t0 = time.perf_counter()
    out = fn()
    _synchronize(dev)
    dt = time.perf_counter() - t0
    peak = _peak_bytes(dev)
    return out, dt, peak


def _rel_fro(a, b) -> float:
    torch = _torch()
    num = torch.linalg.norm((a - b).reshape(-1).to(torch.float64))
    den = torch.linalg.norm(b.reshape(-1).to(torch.float64))
    return float(num / (den if den > 0 else torch.tensor(1.0, dtype=torch.float64)))


def run_once(
    *,
    batch: int = 1,
    heads: int = 8,
    seq: int = 4096,
    dim: int = 64,
    dtype: str = "fp16",
    window: int = 256,
    local_weight: float = 0.85,
    global_weight: float = 0.15,
    freq_decay: float = 1.0,
    gate_strength: float = 0.25,
    temperature: float = 1.0,
    landmarks: int = 64,
    landmark_policy: str = "pooled",
    mode: str = "fixed",
    causal: bool = False,
    seed: int = 0,
    device: str = "auto",
) -> dict:
    torch = _torch()
    from .hybrid import (
        adaptive_hybrid_attention,
        correlation_hybrid_attention,
        hybrid_attention,
        landmark_hybrid_attention,
    )
    from .reference import exact_attention

    if mode not in {"fixed", "adaptive", "corrfft", "landmark", "topk", "both", "all"}:
        raise ValueError("mode must be one of: fixed, adaptive, corrfft, landmark, topk, both, all")

    spec = AttentionSpec(
        batch=batch,
        heads=heads,
        seq=seq,
        dim=dim,
        dtype=dtype,
        window=window,
        local_weight=local_weight,
        global_weight=global_weight,
        freq_decay=freq_decay,
        causal=causal,
        seed=seed,
        device=device,
    )
    dev = resolve_device(spec.device)
    q, k, v = generate_qkv(spec, device=dev)

    # Warm up kernels before measurement.
    _ = exact_attention(q[:, :, : min(seq, 64), :], k[:, :, : min(seq, 64), :], v[:, :, : min(seq, 64), :], causal=causal)
    _ = hybrid_attention(q[:, :, : min(seq, 64), :], k[:, :, : min(seq, 64), :], v[:, :, : min(seq, 64), :],
                         window=min(window, 32), causal=causal, local_weight=local_weight,
                         global_weight=global_weight, freq_decay=freq_decay)
    if mode in {"adaptive", "both", "all"}:
        _ = adaptive_hybrid_attention(
            q[:, :, : min(seq, 64), :],
            k[:, :, : min(seq, 64), :],
            v[:, :, : min(seq, 64), :],
            window=min(window, 32),
            causal=causal,
            local_weight=local_weight,
            global_weight=global_weight,
            freq_decay=freq_decay,
            gate_strength=gate_strength,
        )
    if mode in {"corrfft", "all"}:
        _ = correlation_hybrid_attention(
            q[:, :, : min(seq, 64), :],
            k[:, :, : min(seq, 64), :],
            v[:, :, : min(seq, 64), :],
            window=min(window, 32),
            causal=causal,
            local_weight=local_weight,
            global_weight=global_weight,
            temperature=temperature,
            freq_decay=freq_decay,
        )
    if mode in {"landmark", "topk", "all"}:
        _ = landmark_hybrid_attention(
            q[:, :, : min(seq, 64), :],
            k[:, :, : min(seq, 64), :],
            v[:, :, : min(seq, 64), :],
            window=min(window, 32),
            causal=causal,
            local_weight=local_weight,
            global_weight=global_weight,
            num_landmarks=min(landmarks, min(seq, 64)),
            landmark_policy="topk" if mode == "topk" else landmark_policy,
        )
    _synchronize(dev)

    exact, exact_s, exact_peak = _timed(
        lambda: exact_attention(q, k, v, causal=causal), dev
    )

    candidate_fns = {}
    if mode in {"fixed", "both", "all"}:
        candidate_fns["fixed"] = lambda: hybrid_attention(
            q, k, v,
            window=window,
            causal=causal,
            local_weight=local_weight,
            global_weight=global_weight,
            freq_decay=freq_decay,
        )
    if mode in {"adaptive", "both", "all"}:
        candidate_fns["adaptive"] = lambda: adaptive_hybrid_attention(
            q, k, v,
            window=window,
            causal=causal,
            local_weight=local_weight,
            global_weight=global_weight,
            freq_decay=freq_decay,
            gate_strength=gate_strength,
        )
    if mode in {"corrfft", "all"}:
        candidate_fns["corrfft"] = lambda: correlation_hybrid_attention(
            q, k, v,
            window=window,
            causal=causal,
            local_weight=local_weight,
            global_weight=global_weight,
            temperature=temperature,
            freq_decay=freq_decay,
        )
    if mode in {"landmark", "all"}:
        candidate_fns["landmark"] = lambda: landmark_hybrid_attention(
            q, k, v,
            window=window,
            causal=causal,
            local_weight=local_weight,
            global_weight=global_weight,
            num_landmarks=landmarks,
            landmark_policy=landmark_policy,
        )
    if mode in {"topk", "all"}:
        candidate_fns["topk"] = lambda: landmark_hybrid_attention(
            q, k, v,
            window=window,
            causal=causal,
            local_weight=local_weight,
            global_weight=global_weight,
            num_landmarks=landmarks,
            landmark_policy="topk",
        )

    candidates = {}
    for name, fn in candidate_fns.items():
        out, latency_s, peak = _timed(fn, dev)
        mse = float(torch.mean((out - exact).to(torch.float64) ** 2))
        rel = _rel_fro(out, exact)
        candidates[name] = {
            "latency_s": latency_s,
            "peak_vram_bytes": peak,
            "peak_vram_mib": peak / (1024**2),
            "quality": {
                "mse": mse,
                "rel_frobenius_error": rel,
                "accuracy_proxy": max(0.0, 1.0 - rel),
            },
            "improvement": {
                "faster_than_exact": latency_s < exact_s,
                # VRAM is only measured on CUDA (_peak_bytes returns 0 elsewhere);
                # report None ("unknown") off-CUDA instead of a misleading False.
                "less_vram_than_exact": (peak < exact_peak) if dev.type == "cuda" else None,
                "latency_ratio_exact_over_candidate": (exact_s / latency_s) if latency_s > 0 else math.inf,
            },
        }

    primary = "fixed" if mode in {"fixed", "both", "all"} else mode

    result = {
        "config": {
            **spec.as_dict(),
            "device": str(dev),
            "mode": mode,
            "gate_strength": gate_strength,
            "temperature": temperature,
            "landmarks": landmarks,
            "landmark_policy": landmark_policy,
        },
        "exact": {
            "latency_s": exact_s,
            "peak_vram_bytes": exact_peak,
            "peak_vram_mib": exact_peak / (1024**2),
        },
        "candidates": candidates,
    }
    result["hybrid"] = candidates[primary]
    result["quality"] = candidates[primary]["quality"]
    result["improvement"] = {
        **candidates[primary]["improvement"],
        "latency_ratio_exact_over_hybrid": candidates[primary]["improvement"][
            "latency_ratio_exact_over_candidate"
        ],
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m attention.benchmark",
        description="Benchmark exact attention against the hybrid local+spectral reference.",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp16", "fp32", "fp64"), default="fp16")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--local-weight", type=float, default=0.85)
    parser.add_argument("--global-weight", type=float, default=0.15)
    parser.add_argument("--freq-decay", type=float, default=1.0)
    parser.add_argument("--gate-strength", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--landmarks", type=int, default=64)
    parser.add_argument("--landmark-policy", choices=("pooled", "topk"), default="pooled")
    parser.add_argument("--mode", choices=("fixed", "adaptive", "corrfft", "landmark", "topk", "both", "all"), default="fixed")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    # Invalid knobs (--batch 0, --temperature 0, a bad --landmark-policy, ...) are
    # rejected by AttentionSpec / the hybrid helpers, and a missing GPU / PyTorch
    # raises inside run_once. Report all of these as a clean stderr line and exit
    # 2, matching the matmul and strategy CLIs (see tests/test_cli_errors.py, #175)
    # instead of dumping an uncaught traceback (#201).
    try:
        result = run_once(
            batch=args.batch,
            heads=args.heads,
            seq=args.seq,
            dim=args.dim,
            dtype=args.dtype,
            window=args.window,
            local_weight=args.local_weight,
            global_weight=args.global_weight,
            freq_decay=args.freq_decay,
            gate_strength=args.gate_strength,
            temperature=args.temperature,
            landmarks=args.landmarks,
            landmark_policy=args.landmark_policy,
            mode=args.mode,
            causal=args.causal,
            seed=args.seed,
            device=args.device,
        )
    except (ValueError, RuntimeError, MemoryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"[attention] device={result['config']['device']} seq={result['config']['seq']} "
            f"heads={result['config']['heads']} dim={result['config']['dim']} "
            f"dtype={result['config']['dtype']} window={result['config']['window']} "
            f"mode={result['config']['mode']}"
        )
        print(
            f"[attention] exact  : {result['exact']['latency_s']:.4f}s  "
            f"peak={result['exact']['peak_vram_mib']:.1f} MiB"
        )
        for name, candidate in result["candidates"].items():
            quality = candidate["quality"]
            improvement = candidate["improvement"]
            print(
                f"[attention] {name:<8}: {candidate['latency_s']:.4f}s  "
                f"peak={candidate['peak_vram_mib']:.1f} MiB"
            )
            print(
                f"[attention] {name:<8} quality: mse={quality['mse']:.6e}  "
                f"rel_err={quality['rel_frobenius_error']:.6f}  "
                f"acc_proxy={quality['accuracy_proxy']:.6f}"
            )
            print(
                f"[attention] {name:<8} faster={improvement['faster_than_exact']}  "
                f"less_vram={improvement['less_vram_than_exact']}  "
                f"speedup={improvement['latency_ratio_exact_over_candidate']:.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
