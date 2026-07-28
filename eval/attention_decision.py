"""Pure main-versus-PR decision logic for attention benchmark artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .attention_manifest import DEFAULT_MANIFEST, AttentionManifest, load_manifest


def load_artifact(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _workloads(artifact: dict) -> dict[str, dict]:
    return {item["id"]: item for item in artifact.get("workloads", [])}


def _geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _tier(gain_percent: float, manifest: AttentionManifest) -> str:
    tiers = manifest.raw["decision"].get("tiers_percent") or {
        "L": 20.0,
        "M": 10.0,
        "S": 0.0,
    }
    for name, floor in sorted(tiers.items(), key=lambda item: float(item[1]), reverse=True):
        if gain_percent >= float(floor):
            return name
    return "S"


def compare_artifacts(
    main: dict,
    candidate: dict,
    manifest: AttentionManifest,
    *,
    expected_candidate_commit: str | None = None,
) -> dict:
    reasons = []
    for name, artifact in (("main", main), ("candidate", candidate)):
        if artifact.get("schema_version") != 1:
            reasons.append(f"{name} artifact has unsupported schema")
        if artifact.get("benchmark") != manifest.id:
            reasons.append(f"{name} artifact benchmark does not match manifest")
        if artifact.get("manifest_sha256") != manifest.sha256:
            reasons.append(f"{name} artifact manifest hash does not match")
        if artifact.get("benchmark_contract_sha256") != manifest.benchmark_contract_sha256:
            reasons.append(f"{name} artifact benchmark contract hash does not match")
        if artifact.get("dirty"):
            reasons.append(f"{name} checkout was dirty")
        if manifest.is_active and not artifact.get("merge_eligible"):
            reasons.append(f"{name} artifact is not official/merge-eligible")
    if expected_candidate_commit and candidate.get("commit") != expected_candidate_commit:
        reasons.append("candidate artifact commit does not match queued PR head")

    environment_keys = (
        "gpu_name", "torch", "cuda_runtime", "driver_version", "os", "python",
        "compute_capability", "tf32_matmul", "deterministic_algorithms",
        "cudnn_benchmark", "sdpa_backend_policy", "sdpa_backends_enabled",
    )
    for key in environment_keys:
        if main.get("environment", {}).get(key) != candidate.get("environment", {}).get(key):
            reasons.append(f"environment mismatch between main and candidate: {key}")

    main_by_id = _workloads(main)
    candidate_by_id = _workloads(candidate)
    expected_ids = [workload.id for workload in manifest.workloads]
    if set(main_by_id) != set(expected_ids):
        reasons.append("main artifact workload set does not match manifest")
    if set(candidate_by_id) != set(expected_ids):
        reasons.append("candidate artifact workload set does not match manifest")

    cases = []
    correctness_failed = False
    if not reasons:
        for workload in manifest.workloads:
            base = main_by_id[workload.id]
            pr = candidate_by_id[workload.id]
            passed = bool(pr.get("correctness", {}).get("passed"))
            correctness_failed = correctness_failed or not passed
            base_ms = float(base["candidate"]["timing"]["median_ms"])
            pr_ms = float(pr["candidate"]["timing"]["median_ms"])
            base_vram = int(base["candidate"]["peak_incremental_vram_bytes"])
            pr_vram = int(pr["candidate"]["peak_incremental_vram_bytes"])
            speedup = base_ms / pr_ms if pr_ms > 0 else math.inf
            regression_percent = (pr_ms / base_ms - 1.0) * 100.0 if base_ms > 0 else math.inf
            vram_regression_percent = (
                (pr_vram / base_vram - 1.0) * 100.0
                if base_vram > 0
                else (math.inf if pr_vram > 0 else 0.0)
            )
            cases.append({
                "id": workload.id,
                "mode": workload.mode,
                "scored": workload.scored,
                "correctness_passed": passed,
                "main_median_ms": base_ms,
                "candidate_median_ms": pr_ms,
                "speedup": speedup,
                "latency_regression_percent": regression_percent,
                "main_peak_incremental_vram_bytes": base_vram,
                "candidate_peak_incremental_vram_bytes": pr_vram,
                "vram_regression_percent": vram_regression_percent,
            })

    result = {
        "schema_version": 1,
        "benchmark": manifest.id,
        "manifest_sha256": manifest.sha256,
        "benchmark_contract_sha256": manifest.benchmark_contract_sha256,
        "main_commit": main.get("commit"),
        "candidate_commit": candidate.get("commit"),
        "benchmark_status": manifest.status,
        "cases": cases,
        "tracks": {},
        "verdict": "BLOCKED",
        "label": None,
        "merge": False,
        "reasons": reasons,
    }
    if reasons:
        return result
    if correctness_failed:
        result.update({
            "verdict": "REJECT",
            "label": "eval:REJECT",
            "reasons": ["candidate failed one or more GPU correctness gates"],
        })
        return result

    scored = [case for case in cases if case["scored"]]
    tracks = {}
    for mode in ("prefill", "decode"):
        selected = [case for case in scored if case["mode"] == mode]
        if selected:
            speedup = _geometric_mean([case["speedup"] for case in selected])
            tracks[mode] = {
                "workloads": len(selected),
                "geometric_mean_speedup": speedup,
                "gain_percent": (speedup - 1.0) * 100.0,
                "worst_latency_regression_percent": max(
                    case["latency_regression_percent"] for case in selected
                ),
                "worst_vram_regression_percent": max(
                    case["vram_regression_percent"] for case in selected
                ),
            }
    result["tracks"] = tracks

    # A draft/calibrating benchmark is incapable of authorizing a PR, even if
    # the raw numbers look favorable.  This is the central activation lock.
    if not manifest.is_active:
        result.update({
            "verdict": "CALIBRATION",
            "label": "eval:attention-draft",
            "reasons": [
                "benchmark is draft or calibration/merge lock is still enabled"
            ],
        })
        return result

    decision = manifest.raw["decision"]
    max_latency_regression = float(decision["maximum_per_workload_regression_percent"])
    max_vram_regression = float(decision["maximum_vram_regression_percent"])
    min_gain = float(decision["minimum_speedup_percent"])
    regressed = [
        case["id"] for case in scored
        if case["latency_regression_percent"] > max_latency_regression
        or case["vram_regression_percent"] > max_vram_regression
    ]
    if regressed:
        result.update({
            "verdict": "REJECT",
            "label": "eval:REJECT",
            "reasons": ["protected workload regression: " + ", ".join(regressed)],
        })
        return result

    improving = [track for track, cell in tracks.items() if cell["gain_percent"] >= min_gain]
    if not improving:
        result.update({
            "verdict": "NONE",
            "label": "eval:none",
            "reasons": ["no track improved beyond the calibrated noise floor"],
        })
        return result

    best_gain = max(tracks[track]["gain_percent"] for track in improving)
    tier = _tier(best_gain, manifest)
    result.update({
        "verdict": "ADMIT",
        "label": f"eval:{tier}",
        "merge": True,
        "reasons": [
            f"significant improvement on {', '.join(improving)} with no protected regression"
        ],
    })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.attention_decision",
        description="Compare main and candidate attention benchmark artifacts.",
    )
    parser.add_argument("main_result")
    parser.add_argument("candidate_result")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--expected-candidate-commit")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        result = compare_artifacts(
            load_artifact(args.main_result),
            load_artifact(args.candidate_result),
            manifest,
            expected_candidate_commit=args.expected_candidate_commit,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
