"""Protected RTX 5070 Ti attention-foundation benchmark.

The manifest is deliberately ``draft`` at first.  Draft artifacts are useful
for calibration, but they can never authorize a merge.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import secrets
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .attention_manifest import DEFAULT_MANIFEST, AttentionManifest, load_candidate, load_manifest


DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("attention evaluation requires PyTorch") from exc
    return torch


def _git_value(*args: str, default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip() or default
    except (OSError, subprocess.CalledProcessError):
        return default


def _nvidia_smi(device_index: int) -> dict:
    query = "driver_version,temperature.gpu,power.draw,clocks.sm,memory.total"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi", f"--id={device_index}", f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        driver, temperature, power, clock, memory = [part.strip() for part in output.split(",")]
        return {
            "driver_version": driver,
            "observations": {
                "temperature_c": float(temperature),
                "power_w": float(power),
                "sm_clock_mhz": int(float(clock)),
                "total_vram_mib": float(memory),
            },
        }
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"driver_version": None, "observations": None}


def _environment(device) -> dict:
    torch = _torch()
    props = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    smi = _nvidia_smi(device_index) if device.type == "cuda" else {
        "driver_version": None, "observations": None,
    }
    return {
        "platform": platform.platform(),
        "os": platform.system(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver_version": smi["driver_version"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compute_capability": (
            [props.major, props.minor] if props is not None else None
        ),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if device.type == "cuda" else None,
        "sdpa_backend_policy": "auto",
        "sdpa_backends_enabled": {
            "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
            "memory_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math": bool(torch.backends.cuda.math_sdp_enabled()),
        } if device.type == "cuda" else None,
        "observations": smi["observations"],
    }


def environment_mismatches(manifest: AttentionManifest, environment: dict) -> list[str]:
    expected = manifest.raw["era"]
    checks = {
        "gpu_name": environment.get("gpu_name"),
        "os": environment.get("os"),
        "python": environment.get("python"),
        "torch": environment.get("torch"),
    }
    mismatches = []
    for name, actual in checks.items():
        wanted = str(expected.get(name, ""))
        if name == "python":
            actual = ".".join(str(actual).split(".")[:2])
        if str(actual) != wanted:
            mismatches.append(f"{name}: expected {wanted!r}, got {actual!r}")
    if bool(environment.get("tf32_matmul")) != bool(expected.get("tf32")):
        mismatches.append(
            f"tf32: expected {bool(expected.get('tf32'))}, "
            f"got {bool(environment.get('tf32_matmul'))}"
        )
    for name in ("deterministic_algorithms", "cudnn_benchmark"):
        if bool(environment.get(name)) != bool(expected.get(name)):
            mismatches.append(
                f"{name}: expected {bool(expected.get(name))}, "
                f"got {bool(environment.get(name))}"
            )
    return mismatches


def _summary(samples_ms: list[float]) -> dict:
    if not samples_ms:
        raise ValueError("timing samples cannot be empty")
    ordered = sorted(float(value) for value in samples_ms)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "samples_ms": ordered,
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[p90_index],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def merge_artifact_shards(shards: list[dict]) -> dict:
    """Combine balanced timing shards from one checkout into one artifact."""
    if not shards:
        raise ValueError("at least one benchmark shard is required")
    first = shards[0]
    integrity_fields = (
        "fresh_inputs_per_call",
        "validate_timed_outputs",
        "candidate_import_after_reference",
        "runtime_state_guard",
    )
    for shard in shards:
        measurement = shard.get("measurement") or {}
        if measurement.get("seed_policy") != "os-random-per-call-published-after":
            raise ValueError("benchmark shard has an unsupported seed policy")
        if any(measurement.get(field) is not True for field in integrity_fields):
            raise ValueError("benchmark shard is missing an integrity guarantee")
    identity = (
        "schema_version", "benchmark", "manifest_sha256",
        "benchmark_contract_sha256", "candidate", "commit", "dirty", "seed",
    )
    for shard in shards[1:]:
        if any(shard.get(key) != first.get(key) for key in identity):
            raise ValueError("benchmark shards do not have the same identity")
        first_environment = dict(first.get("environment") or {})
        shard_environment = dict(shard.get("environment") or {})
        first_environment.pop("observations", None)
        shard_environment.pop("observations", None)
        if shard_environment != first_environment:
            raise ValueError("benchmark shard environments do not match")
    by_shard = [{item["id"]: item for item in shard["workloads"]} for shard in shards]
    if any(set(cells) != set(by_shard[0]) for cells in by_shard[1:]):
        raise ValueError("benchmark shard workload sets do not match")

    merged = dict(first)
    merged["environment"] = dict(first.get("environment") or {})
    merged["environment"]["observations_by_shard"] = [
        shard.get("environment", {}).get("observations") for shard in shards
    ]
    merged_workloads = []
    for item in first["workloads"]:
        cells = [cells[item["id"]] for cells in by_shard]
        output = dict(item)
        output["input_seeds"] = {
            "correctness": [cell["input_seeds"]["correctness"] for cell in cells],
            "production": {
                "warmups": [
                    seed
                    for cell in cells
                    for seed in cell["input_seeds"]["production"]["warmups"]
                ],
                "measured": [
                    seed
                    for cell in cells
                    for seed in cell["input_seeds"]["production"]["measured"]
                ],
            },
            "candidate": {
                "warmups": [
                    seed
                    for cell in cells
                    for seed in cell["input_seeds"]["candidate"]["warmups"]
                ],
                "measured": [
                    seed
                    for cell in cells
                    for seed in cell["input_seeds"]["candidate"]["measured"]
                ],
            },
        }
        output["correctness"] = dict(item["correctness"])
        output["correctness"]["passed"] = all(
            bool(cell["correctness"]["passed"]) for cell in cells
        )
        timed = [cell["correctness"].get("timed_output_validation") for cell in cells]
        if all(check is not None for check in timed):
            output["correctness"]["timed_output_validation"] = {
                "passed": all(bool(check["passed"]) for check in timed),
                "repetitions": sum(int(check["repetitions"]) for check in timed),
                "maximum_relative_frobenius_error": max(
                    float(check["maximum_relative_frobenius_error"]) for check in timed
                ),
                "maximum_absolute_error": max(
                    float(check["maximum_absolute_error"]) for check in timed
                ),
                "finite": all(bool(check["finite"]) for check in timed),
            }
        for implementation in ("production", "candidate"):
            samples = []
            for cell in cells:
                samples.extend(cell[implementation]["timing"]["samples_ms"])
            output[implementation] = {
                "timing": _summary(samples),
                "peak_incremental_vram_bytes": max(
                    int(cell[implementation]["peak_incremental_vram_bytes"])
                    for cell in cells
                ),
            }
        merged_workloads.append(output)
    merged["workloads"] = merged_workloads
    merged["measurement"] = {
        "warmups_per_shard": first["measurement"]["warmups"],
        "repetitions": sum(int(shard["measurement"]["repetitions"]) for shard in shards),
        "shards": len(shards),
        "seed_policy": "os-random-per-call-published-after",
        "fresh_inputs_per_call": True,
        "validate_timed_outputs": True,
        "candidate_import_after_reference": True,
        "runtime_state_guard": True,
    }
    merged.pop("shard", None)
    return merged


def _quality(candidate, reference) -> dict:
    torch = _torch()
    candidate64 = candidate.to(torch.float64)
    reference64 = reference.to(torch.float64)
    delta = candidate64 - reference64
    denominator = torch.linalg.vector_norm(reference64.reshape(-1))
    relative = torch.linalg.vector_norm(delta.reshape(-1)) / torch.clamp(denominator, min=1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        candidate64.reshape(1, -1), reference64.reshape(1, -1), dim=1
    )
    return {
        "relative_frobenius_error": float(relative),
        "maximum_absolute_error": float(delta.abs().max()),
        "mean_squared_error": float(torch.mean(delta * delta)),
        "cosine_similarity": float(cosine[0]),
        "finite": bool(torch.isfinite(candidate).all()),
    }


def _explicit_oracle(q, k, v, *, causal: bool):
    torch = _torch()
    q32, k32, v32 = q.float(), k.float(), v.float()
    scores = torch.matmul(q32, k32.transpose(-1, -2)) / math.sqrt(float(q.shape[-1]))
    if causal:
        q_len, kv_len = q.shape[-2], k.shape[-2]
        if q_len != kv_len:
            raise ValueError("explicit causal oracle requires equal query and KV lengths")
        mask = torch.ones((q_len, kv_len), dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v32)


def _runtime_state(torch) -> tuple:
    """Snapshot process-wide state contributor code must restore before returning."""
    return (
        torch.nn.functional.scaled_dot_product_attention,
        torch.cuda.Event,
        torch.cuda.synchronize,
        torch.cuda.memory_allocated,
        torch.cuda.reset_peak_memory_stats,
        torch.cuda.max_memory_allocated,
        torch.nn.functional.cosine_similarity,
        torch.linalg.vector_norm,
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.are_deterministic_algorithms_enabled()),
        bool(torch.backends.cudnn.benchmark),
        bool(torch.backends.cudnn.deterministic),
        bool(torch.backends.cuda.flash_sdp_enabled()),
        bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        bool(torch.backends.cuda.math_sdp_enabled()),
        torch.get_default_dtype(),
    )


def _assert_runtime_state(torch, expected: tuple) -> None:
    if _runtime_state(torch) != expected:
        raise RuntimeError(
            "candidate modified protected PyTorch/CUDA runtime state without restoring it"
        )


def _timed_validation_summary(qualities: list[dict], tolerances: dict) -> dict:
    if not qualities:
        raise ValueError("timed output validation requires at least one result")
    maximum_relative = max(float(item["relative_frobenius_error"]) for item in qualities)
    maximum_absolute = max(float(item["maximum_absolute_error"]) for item in qualities)
    finite = all(bool(item["finite"]) for item in qualities)
    return {
        "passed": bool(
            finite
            and maximum_relative <= tolerances["max_relative_frobenius_error"]
            and maximum_absolute <= tolerances["max_absolute_error"]
        ),
        "repetitions": len(qualities),
        "maximum_relative_frobenius_error": maximum_relative,
        "maximum_absolute_error": maximum_absolute,
        "finite": finite,
    }


def _time_fresh_inputs(
    fn,
    *,
    workload,
    seed_source,
    input_factory,
    quality_fn,
    validation_summary_fn,
    device,
    warmups: int,
    repetitions: int,
    protected_reference=None,
    tolerances: dict | None = None,
    runtime_guard=None,
) -> tuple[dict, int, dict | None, dict]:
    """Time one call per fresh input and optionally validate every timed output.

    Input creation and protected replay are outside CUDA events.  A candidate
    therefore cannot win by caching the correctness output or one repeated
    tensor pointer, while the measured value remains operator-only latency.
    """
    torch = _torch()
    event_factory = torch.cuda.Event
    synchronize = torch.cuda.synchronize
    memory_allocated = torch.cuda.memory_allocated
    reset_peak = torch.cuda.reset_peak_memory_stats
    max_memory_allocated = torch.cuda.max_memory_allocated

    warmup_seeds = []
    measured_seeds = []
    for _ in range(warmups):
        seed = int(seed_source())
        warmup_seeds.append(seed)
        q, k, v = input_factory(workload, device=device, seed=seed)
        output = fn(q, k, v, causal=workload.causal)
        if runtime_guard is not None:
            runtime_guard()
        del q, k, v, output
    synchronize(device)

    samples = []
    peak_increments = []
    timed_qualities = []
    for _ in range(repetitions):
        seed = int(seed_source())
        measured_seeds.append(seed)
        q, k, v = input_factory(workload, device=device, seed=seed)
        synchronize(device)
        resident = int(memory_allocated(device))
        reset_peak(device)
        start = event_factory(enable_timing=True)
        end = event_factory(enable_timing=True)
        start.record()
        output = fn(q, k, v, causal=workload.causal)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        peak_increments.append(max(0, int(max_memory_allocated(device)) - resident))
        if runtime_guard is not None:
            runtime_guard()
        if protected_reference is not None:
            reference_q, reference_k, reference_v = input_factory(
                workload, device=device, seed=seed
            )
            reference_output = protected_reference(
                reference_q, reference_k, reference_v, causal=workload.causal
            )
            timed_qualities.append(quality_fn(output, reference_output))
            del reference_q, reference_k, reference_v, reference_output
        del q, k, v, output
    synchronize(device)
    validation = None
    if protected_reference is not None:
        if tolerances is None:
            raise ValueError("timed validation requires correctness tolerances")
        validation = validation_summary_fn(timed_qualities, tolerances)
    return (
        _summary(samples),
        max(peak_increments),
        validation,
        {"warmups": warmup_seeds, "measured": measured_seeds},
    )


def _inputs(workload, *, device, seed: int):
    torch = _torch()
    dtype = getattr(torch, DTYPES[workload.dtype])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    prefix = (workload.batch, workload.heads)
    storage_dim = workload.dim * 2 if workload.layout == "noncontiguous" else workload.dim
    q = torch.randn((*prefix, workload.q_len, storage_dim), device=device, dtype=dtype, generator=generator)
    k = torch.randn((*prefix, workload.kv_len, storage_dim), device=device, dtype=dtype, generator=generator)
    v = torch.randn((*prefix, workload.kv_len, storage_dim), device=device, dtype=dtype, generator=generator)
    if workload.layout == "noncontiguous":
        q, k, v = q[..., ::2], k[..., ::2], v[..., ::2]
    return q, k, v


def run_benchmark(
    manifest: AttentionManifest,
    *,
    candidate_spec: str | None = None,
    device_index: int | None = None,
    seed: int | None = None,
    repetitions: int | None = None,
    warmups: int | None = None,
    official: bool = False,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the attention-foundation benchmark")
    device_index = manifest.raw["era"]["device"] if device_index is None else device_index
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = bool(manifest.raw["era"].get("tf32", False))
    torch.use_deterministic_algorithms(
        bool(manifest.raw["era"].get("deterministic_algorithms", False))
    )
    torch.backends.cudnn.benchmark = bool(manifest.raw["era"].get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = False

    if seed is None:
        seed = secrets.randbits(63)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    environment = _environment(device)
    mismatches = environment_mismatches(manifest, environment)
    if official and mismatches:
        raise RuntimeError("official environment mismatch: " + "; ".join(mismatches))

    measurement = manifest.raw["measurement"]
    requested_repetitions = repetitions
    warmups = measurement["warmups"] if warmups is None else warmups
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard_index and shard_count must be provided together")
    if shard_count is not None:
        if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
            raise ValueError("invalid benchmark shard")
        if requested_repetitions is None:
            if measurement["repetitions"] % shard_count:
                raise ValueError("manifest repetitions must divide evenly into shards")
            repetitions = measurement["repetitions"] // shard_count
        else:
            repetitions = requested_repetitions
    else:
        repetitions = (
            measurement["repetitions"]
            if requested_repetitions is None
            else requested_repetitions
        )
    if official:
        official_repetitions = repetitions * (shard_count or 1)
        if official_repetitions != measurement["repetitions"] or warmups != measurement["warmups"]:
            raise ValueError("official runs cannot change total manifest measurement counts")
    if repetitions <= 0 or warmups <= 0:
        raise ValueError("warmups and repetitions must be positive")

    candidate_spec = candidate_spec or manifest.raw["candidate"]
    tolerances = manifest.raw["correctness"]
    protected_sdpa = torch.nn.functional.scaled_dot_product_attention
    protected_inputs = _inputs
    protected_quality = _quality
    protected_validation_summary = _timed_validation_summary
    protected_assert_runtime = _assert_runtime_state

    def protected_reference(q, k, v, *, causal: bool):
        return protected_sdpa(q, k, v, dropout_p=0.0, is_causal=causal)

    expected_runtime = _runtime_state(torch)
    protected_randbits = secrets.SystemRandom().getrandbits
    production_records = {}

    # Contributor code is deliberately not imported until every protected
    # reference timing and affordable explicit-oracle output has been captured.
    for workload in manifest.workloads:
        correctness_seed = protected_randbits(63)
        with torch.inference_mode():
            q, k, v = protected_inputs(workload, device=device, seed=correctness_seed)
            production_output = protected_reference(q, k, v, causal=workload.causal)
            oracle_output = None
            production_oracle_quality = None
            if workload.oracle:
                oracle_output = _explicit_oracle(q, k, v, causal=workload.causal)
                production_oracle_quality = protected_quality(production_output, oracle_output)
            production_timing, production_peak, _, production_seeds = _time_fresh_inputs(
                protected_reference,
                workload=workload,
                seed_source=lambda: protected_randbits(63),
                input_factory=protected_inputs,
                quality_fn=protected_quality,
                validation_summary_fn=protected_validation_summary,
                device=device,
                warmups=warmups,
                repetitions=repetitions,
            )
        production_records[workload.id] = {
            "timing": production_timing,
            "peak_incremental_vram_bytes": production_peak,
            "oracle_output": oracle_output,
            "oracle_quality": production_oracle_quality,
            "correctness_seed": correctness_seed,
            "timing_seeds": production_seeds,
        }
        del q, k, v, production_output
        torch.cuda.empty_cache()

    candidate = load_candidate(candidate_spec)
    protected_assert_runtime(torch, expected_runtime)

    def runtime_guard():
        protected_assert_runtime(torch, expected_runtime)

    results = []
    for workload in manifest.workloads:
        correctness_seed = production_records[workload.id]["correctness_seed"]
        q, k, v = protected_inputs(workload, device=device, seed=correctness_seed)
        with torch.inference_mode():
            production_output = protected_reference(q, k, v, causal=workload.causal)
            candidate_output = candidate(q, k, v, causal=workload.causal)
            runtime_guard()
            candidate_quality = protected_quality(candidate_output, production_output)
            oracle_quality = None
            if workload.oracle:
                oracle_output = production_records[workload.id]["oracle_output"]
                oracle_quality = {
                    "production": production_records[workload.id]["oracle_quality"],
                    "candidate": protected_quality(candidate_output, oracle_output),
                }
            oracle_candidate = (
                oracle_quality["candidate"] if oracle_quality is not None else None
            )
            causal_prefix = None
            if workload.causal:
                split = max(1, workload.q_len // 2)
                changed_k = k.clone()
                changed_v = v.clone()
                changed_k[..., split:, :] += 3.0
                changed_v[..., split:, :] -= 3.0
                changed_output = candidate(q, changed_k, changed_v, causal=True)
                runtime_guard()
                prefix_quality = protected_quality(
                    changed_output[..., :split, :],
                    candidate_output[..., :split, :],
                )
                causal_prefix = {
                    "split": split,
                    "passed": (
                        prefix_quality["finite"]
                        and prefix_quality["maximum_absolute_error"] <= 1e-3
                    ),
                    "quality": prefix_quality,
                }
                del changed_k, changed_v, changed_output
            (
                candidate_timing,
                candidate_peak,
                timed_validation,
                candidate_seeds,
            ) = _time_fresh_inputs(
                candidate,
                workload=workload,
                seed_source=lambda: protected_randbits(63),
                input_factory=protected_inputs,
                quality_fn=protected_quality,
                validation_summary_fn=protected_validation_summary,
                device=device,
                warmups=warmups,
                repetitions=repetitions,
                protected_reference=protected_reference,
                tolerances=tolerances,
                runtime_guard=runtime_guard,
            )
            passed = (
                candidate_quality["finite"]
                and candidate_quality["relative_frobenius_error"]
                <= tolerances["max_relative_frobenius_error"]
                and candidate_quality["maximum_absolute_error"]
                <= tolerances["max_absolute_error"]
                and (
                    oracle_candidate is None
                    or (
                        oracle_candidate["finite"]
                        and oracle_candidate["relative_frobenius_error"]
                        <= tolerances["max_relative_frobenius_error"]
                        and oracle_candidate["maximum_absolute_error"]
                        <= tolerances["max_absolute_error"]
                    )
                )
                and (causal_prefix is None or causal_prefix["passed"])
                and timed_validation["passed"]
            )
        production_record = production_records[workload.id]
        results.append({
            "id": workload.id,
            "mode": workload.mode,
            "scored": workload.scored,
            "shape": {
                "batch": workload.batch,
                "heads": workload.heads,
                "q_len": workload.q_len,
                "kv_len": workload.kv_len,
                "dim": workload.dim,
                "dtype": workload.dtype,
                "causal": workload.causal,
                "layout": workload.layout,
            },
            "correctness": {
                "passed": passed,
                "candidate_vs_production": candidate_quality,
                "oracle": oracle_quality,
                "causal_prefix_invariance": causal_prefix,
                "timed_output_validation": timed_validation,
            },
            "input_seeds": {
                "correctness": correctness_seed,
                "production": production_record["timing_seeds"],
                "candidate": candidate_seeds,
            },
            "production": {
                "timing": production_record["timing"],
                "peak_incremental_vram_bytes": production_record["peak_incremental_vram_bytes"],
            },
            "candidate": {
                "timing": candidate_timing,
                "peak_incremental_vram_bytes": candidate_peak,
            },
        })
        if production_record["oracle_output"] is not None:
            del production_record["oracle_output"]
        del q, k, v, production_output, candidate_output
        torch.cuda.empty_cache()

    commit = _git_value("rev-parse", "HEAD")
    dirty = bool(_git_value("status", "--porcelain", default=""))
    eligible_environment = not mismatches and not dirty
    result = {
        "schema_version": int(manifest.raw.get("result_schema_version", 1)),
        "benchmark": manifest.id,
        "benchmark_status": manifest.status,
        "manifest_sha256": manifest.sha256,
        "benchmark_contract_sha256": manifest.benchmark_contract_sha256,
        "candidate": candidate_spec,
        "commit": commit,
        "dirty": dirty,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seed": seed,
        "official_requested": official,
        "official_environment": eligible_environment,
        "merge_eligible": bool(manifest.is_active and official and eligible_environment),
        "environment_mismatches": mismatches,
        "environment": environment,
        "measurement": {
            "warmups": warmups,
            "repetitions": repetitions,
            "seed_policy": "os-random-per-call-published-after",
            "fresh_inputs_per_call": True,
            "validate_timed_outputs": True,
            "candidate_import_after_reference": True,
            "runtime_state_guard": True,
        },
        "workloads": results,
    }
    if shard_count is not None:
        result["shard"] = {"index": shard_index, "count": shard_count}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.attention_benchmark",
        description="Run the protected attention-foundation workload in one checkout.",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--candidate")
    parser.add_argument("--device", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        help="run-correlation nonce; input seeds are generated independently per call",
    )
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        result = run_benchmark(
            manifest,
            candidate_spec=args.candidate,
            device_index=args.device,
            seed=args.seed,
            repetitions=args.repetitions,
            warmups=args.warmups,
            official=args.official,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    if args.json or not args.output:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
