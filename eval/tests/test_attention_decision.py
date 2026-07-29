from __future__ import annotations

import json

from eval.attention_decision import compare_artifacts
from eval.attention_manifest import DEFAULT_MANIFEST, benchmark_contract_sha256, load_manifest


def _artifact(manifest, *, commit: str, factor: float = 1.0, correct: bool = True):
    environment = {
        "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
        "torch": "2.11.0+cu128",
        "cuda_runtime": "12.8",
        "os": "Windows",
        "python": "3.12.12",
    }
    workloads = []
    for index, workload in enumerate(manifest.workloads):
        median = (1.0 + index) * factor
        seed_base = index * 1000
        cell = {
            "timing": {"median_ms": median},
            "peak_incremental_vram_bytes": 1000,
        }
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
                "passed": correct,
                "timed_output_validation": {
                    "passed": correct,
                    "repetitions": 30,
                },
            },
            "candidate": cell,
        })
    return {
        "schema_version": manifest.raw.get("result_schema_version", 1),
        "benchmark": manifest.id,
        "manifest_sha256": manifest.sha256,
        "benchmark_contract_sha256": manifest.benchmark_contract_sha256,
        "candidate": manifest.raw["candidate"],
        "commit": commit,
        "seed": 1234,
        "dirty": False,
        "merge_eligible": manifest.is_active,
        "measurement": {
            "repetitions": 30,
            "seed_policy": "os-random-per-call-published-after",
            "fresh_inputs_per_call": True,
            "validate_timed_outputs": True,
            "candidate_import_after_reference": True,
            "runtime_state_guard": True,
        },
        "environment": environment,
        "workloads": workloads,
    }


def _frozen_manifest(tmp_path):
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    raw["status"] = "frozen"
    raw["decision"].update({
        "calibration_required": False,
        "minimum_speedup_percent": 5.0,
        "maximum_per_workload_regression_percent": 3.0,
        "maximum_vram_regression_percent": 2.0,
        "merge_enabled": True,
        "tiers_percent": {"L": 20.0, "M": 10.0, "S": 5.0},
    })
    raw["correctness"]["calibration_required"] = False
    raw["calibration"] = {
        "sessions": 3,
        "benchmark_contract_sha256": benchmark_contract_sha256(raw),
    }
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_manifest(path)


def _draft_manifest(tmp_path):
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    raw["status"] = "draft"
    raw["decision"]["calibration_required"] = True
    raw["decision"]["merge_enabled"] = False
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_manifest(path)


def test_draft_manifest_can_never_authorize_merge(tmp_path):
    manifest = _draft_manifest(tmp_path)
    result = compare_artifacts(
        _artifact(manifest, commit="main"),
        _artifact(manifest, commit="candidate", factor=0.5),
        manifest,
    )
    assert result["verdict"] == "CALIBRATION"
    assert not result["merge"]


def test_incorrect_candidate_is_rejected_before_performance():
    manifest = load_manifest()
    result = compare_artifacts(
        _artifact(manifest, commit="main"),
        _artifact(manifest, commit="candidate", factor=0.5, correct=False),
        manifest,
    )
    assert result["verdict"] == "REJECT"
    assert result["label"] == "eval:REJECT"


def test_active_manifest_admits_significant_no_regression_gain(tmp_path):
    manifest = _frozen_manifest(tmp_path)
    result = compare_artifacts(
        _artifact(manifest, commit="main"),
        _artifact(manifest, commit="candidate", factor=0.8),
        manifest,
        expected_candidate_commit="candidate",
    )
    assert result["verdict"] == "ADMIT"
    assert result["label"] == "eval:L"
    assert result["merge"]


def test_active_manifest_rejects_one_protected_regression(tmp_path):
    manifest = _frozen_manifest(tmp_path)
    main = _artifact(manifest, commit="main")
    candidate = _artifact(manifest, commit="candidate", factor=0.8)
    candidate["workloads"][0]["candidate"]["timing"]["median_ms"] = 2.0
    result = compare_artifacts(main, candidate, manifest)
    assert result["verdict"] == "REJECT"
    assert "protected workload regression" in result["reasons"][0]


def test_active_manifest_rejects_one_protected_vram_regression(tmp_path):
    manifest = _frozen_manifest(tmp_path)
    main = _artifact(manifest, commit="main")
    candidate = _artifact(manifest, commit="candidate", factor=0.8)
    candidate["workloads"][0]["candidate"]["peak_incremental_vram_bytes"] = 2000
    result = compare_artifacts(main, candidate, manifest)
    assert result["verdict"] == "REJECT"
    assert "protected workload regression" in result["reasons"][0]


def test_active_manifest_returns_none_inside_noise_floor(tmp_path):
    manifest = _frozen_manifest(tmp_path)
    result = compare_artifacts(
        _artifact(manifest, commit="main"),
        _artifact(manifest, commit="candidate", factor=0.99),
        manifest,
    )
    assert result["verdict"] == "NONE"
    assert result["label"] == "eval:none"


def test_candidate_commit_must_match_queue_head():
    manifest = load_manifest()
    result = compare_artifacts(
        _artifact(manifest, commit="main"),
        _artifact(manifest, commit="candidate"),
        manifest,
        expected_candidate_commit="different",
    )
    assert result["verdict"] == "BLOCKED"
    assert "queued PR head" in result["reasons"][0]


def test_repeated_timing_input_seed_blocks_decision():
    manifest = load_manifest()
    main = _artifact(manifest, commit="main")
    candidate = _artifact(manifest, commit="candidate")
    candidate["workloads"][0]["input_seeds"]["candidate"]["measured"] = [7] * 30
    result = compare_artifacts(main, candidate, manifest)
    assert result["verdict"] == "BLOCKED"
    assert any("not unique" in reason for reason in result["reasons"])
