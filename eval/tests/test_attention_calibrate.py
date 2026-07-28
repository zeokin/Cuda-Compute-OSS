from __future__ import annotations

import pytest

from eval.attention_calibrate import calibrate
from eval.attention_manifest import load_manifest


def _artifact(manifest, *, factor: float = 1.0, commit: str = "main"):
    workloads = []
    for index, workload in enumerate(manifest.workloads):
        workloads.append({
            "id": workload.id,
            "candidate": {
                "timing": {"median_ms": (index + 1) * factor},
                "peak_incremental_vram_bytes": 1000,
            },
        })
    return {
        "benchmark": manifest.id,
        "manifest_sha256": manifest.sha256,
        "commit": commit,
        "candidate": manifest.raw["candidate"],
        "dirty": False,
        "official_requested": True,
        "official_environment": True,
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
    stale["manifest_sha256"] = "0" * 64
    report = calibrate([stale, stale, stale], manifest)
    assert not report["ready"]
    assert all("manifest hash" in reason for reason in report["reasons"])
