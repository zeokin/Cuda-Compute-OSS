"""Derive draft attention decision recommendations from clean main sessions.

This tool never edits or activates the manifest. It produces an auditable
calibration report for maintainer review before thresholds are frozen.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from .attention_decision import load_artifact
from .attention_manifest import DEFAULT_MANIFEST, AttentionManifest, load_manifest


def _cells(artifact: dict) -> dict[str, dict]:
    return {cell["id"]: cell for cell in artifact.get("workloads", [])}


def _percent_delta(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return math.inf
    return abs(a / b - 1.0) * 100.0


def calibrate(artifacts: list[dict], manifest: AttentionManifest) -> dict:
    if len(artifacts) < 3:
        raise ValueError("calibration requires at least three clean session artifacts")
    reasons = []
    identity_keys = ("benchmark", "benchmark_contract_sha256", "commit", "candidate")
    first = artifacts[0]
    for index, artifact in enumerate(artifacts):
        if artifact.get("benchmark") != manifest.id:
            reasons.append(f"session {index + 1} benchmark does not match the manifest")
        if artifact.get("benchmark_contract_sha256") != manifest.benchmark_contract_sha256:
            reasons.append(f"session {index + 1} benchmark contract hash does not match")
        if any(artifact.get(key) != first.get(key) for key in identity_keys):
            reasons.append(f"session {index + 1} identity does not match session 1")
        if artifact.get("dirty"):
            reasons.append(f"session {index + 1} used a dirty checkout")
        if not artifact.get("official_requested") or not artifact.get("official_environment"):
            reasons.append(f"session {index + 1} is not an official-environment run")
        invariant_environment = (
            "gpu_name", "torch", "cuda_runtime", "driver_version", "os", "python",
            "compute_capability", "tf32_matmul", "deterministic_algorithms",
            "cudnn_benchmark", "sdpa_backend_policy", "sdpa_backends_enabled",
        )
        if any(
            artifact.get("environment", {}).get(key)
            != first.get("environment", {}).get(key)
            for key in invariant_environment
        ):
            reasons.append(f"session {index + 1} invariant environment differs from session 1")
    expected_ids = {workload.id for workload in manifest.workloads}
    session_cells = [_cells(artifact) for artifact in artifacts]
    if any(set(cells) != expected_ids for cells in session_cells):
        reasons.append("one or more sessions do not contain the complete manifest workload")
    if reasons:
        return {
            "schema_version": 1,
            "benchmark": manifest.id,
            "ready": False,
            "reasons": reasons,
            "workloads": [],
        }

    workload_reports = []
    maximum_pairwise_latency_delta = 0.0
    maximum_pairwise_vram_delta = 0.0
    for workload in manifest.workloads:
        cells = [session[workload.id] for session in session_cells]
        medians = [float(cell["candidate"]["timing"]["median_ms"]) for cell in cells]
        peaks = [float(cell["candidate"]["peak_incremental_vram_bytes"]) for cell in cells]
        latency_deltas = [
            _percent_delta(a, b) for a, b in itertools.combinations(medians, 2)
        ]
        vram_deltas = [
            _percent_delta(a, b) if a > 0 and b > 0 else (0.0 if a == b else math.inf)
            for a, b in itertools.combinations(peaks, 2)
        ]
        max_latency = max(latency_deltas, default=0.0)
        max_vram = max(vram_deltas, default=0.0)
        maximum_pairwise_latency_delta = max(maximum_pairwise_latency_delta, max_latency)
        maximum_pairwise_vram_delta = max(maximum_pairwise_vram_delta, max_vram)
        workload_reports.append({
            "id": workload.id,
            "scored": workload.scored,
            "session_medians_ms": medians,
            "median_of_session_medians_ms": statistics.median(medians),
            "maximum_pairwise_latency_delta_percent": max_latency,
            "session_peak_incremental_vram_bytes": peaks,
            "maximum_pairwise_vram_delta_percent": max_vram,
        })

    minimum_speedup = max(5.0, 2.0 * maximum_pairwise_latency_delta)
    max_latency_regression = max(3.0, 2.0 * maximum_pairwise_latency_delta)
    max_vram_regression = max(2.0, 2.0 * maximum_pairwise_vram_delta)
    ready = maximum_pairwise_latency_delta <= 5.0 and math.isfinite(max_vram_regression)
    report_reasons = [] if ready else [
        "baseline repeatability exceeds the 5% latency budget or VRAM is unstable"
    ]
    return {
        "schema_version": 1,
        "benchmark": manifest.id,
        "manifest_sha256": manifest.sha256,
        "benchmark_contract_sha256": manifest.benchmark_contract_sha256,
        "commit": first.get("commit"),
        "sessions": len(artifacts),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "ready": ready,
        "reasons": report_reasons,
        "maximum_pairwise_latency_delta_percent": maximum_pairwise_latency_delta,
        "maximum_pairwise_vram_delta_percent": maximum_pairwise_vram_delta,
        "recommended_decision": {
            "minimum_speedup_percent": minimum_speedup,
            "maximum_per_workload_regression_percent": max_latency_regression,
            "maximum_vram_regression_percent": max_vram_regression,
            "tiers_percent": {
                "S": minimum_speedup,
                "M": max(10.0, 2.0 * minimum_speedup),
                "L": max(20.0, 4.0 * minimum_speedup),
            },
        },
        "workloads": workload_reports,
        "note": "recommendations require maintainer review and certification; this tool never activates the manifest",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.attention_calibrate",
        description="Build a noise/repeatability report from clean main artifacts.",
    )
    parser.add_argument("results", nargs="+")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        report = calibrate([load_artifact(path) for path in args.results], manifest)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
