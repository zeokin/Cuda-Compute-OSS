from __future__ import annotations

import pytest

from eval.attention_calibrate import calibrate
from eval.attention_manifest import load_manifest


def _artifact(manifest, *, factor: float = 1.0, commit: str = "main"):
    workloads = []
    for index, workload in enumerate(manifest.workloads):
        seed_base = index * 1000
        workloads.append({
            "id": workload.id,
            "input_seeds": {
                "correctness": seed_base,
                "production": {
                    "warmups": [seed_base + 1],
                    "measured": list(range(seed_base + 2, seed_base + 32)),
                },
                "candidate": {
                    "warmups": [seed_base + 32],
                    "measured": list(range(seed_base + 33, seed_base + 63)),
                },
            },
            "correctness": {
                "passed": True,
                "timed_output_validation": {"passed": True},
            },
            "candidate": {
                "timing": {"median_ms": (index + 1) * factor},
                "peak_incremental_vram_bytes": 1000,
            },
        })
    return {
        "schema_version": manifest.raw.get("result_schema_version", 1),
        "benchmark": manifest.id,
        "manifest_sha256": manifest.sha256,
        "benchmark_contract_sha256": manifest.benchmark_contract_sha256,
        "commit": commit,
        "candidate": manifest.raw["candidate"],
        "dirty": False,
        "seed": 1000 + int(factor * 100),
        "official_requested": True,
        "official_environment": True,
        "measurement": {
            "repetitions": 30,
            "seed_policy": "os-random-per-call-published-after",
            "fresh_inputs_per_call": True,
            "validate_timed_outputs": True,
            "candidate_import_after_reference": True,
            "runtime_state_guard": True,
        },
        "environment": {"gpu": "5070ti"},
        "workloads": workloads,
    }


def test_calibration_requires_three_sessions():
    manifest = load_manifest()
    with pytest.raises(ValueError, match="three"):
        calibrate([_artifact(manifest)], manifest)


def test_stable_sessions_produce_conservative_recommendations():
    manifest = load_manifest()
    report = calibrate([
        _artifact(manifest, factor=1.00),
        _artifact(manifest, factor=1.005),
        _artifact(manifest, factor=0.995),
    ], manifest)
    assert report["ready"]
    assert report["recommended_decision"]["minimum_speedup_percent"] == 5.0
    assert report["recommended_decision"]["maximum_per_workload_regression_percent"] == 3.0
    assert report["recommended_decision"]["maximum_vram_regression_percent"] == 2.0


def test_unstable_sessions_do_not_calibrate():
    manifest = load_manifest()
    report = calibrate([
        _artifact(manifest, factor=1.00),
        _artifact(manifest, factor=1.10),
        _artifact(manifest, factor=0.90),
    ], manifest)
    assert not report["ready"]


def test_mismatched_commit_is_blocked():
    manifest = load_manifest()
    report = calibrate([
        _artifact(manifest),
        _artifact(manifest),
        _artifact(manifest, commit="different"),
    ], manifest)
    assert not report["ready"]
    assert "identity" in report["reasons"][0]


def test_stale_manifest_hash_is_blocked():
    manifest = load_manifest()
    stale = _artifact(manifest)
    stale["benchmark_contract_sha256"] = "0" * 64
    report = calibrate([stale, stale, stale], manifest)
    assert not report["ready"]
    assert all("benchmark contract hash" in reason for reason in report["reasons"])
