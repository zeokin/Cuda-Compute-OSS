from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eval.attention_manifest import (
    DEFAULT_MANIFEST,
    benchmark_contract_sha256,
    load_manifest,
    manifest_sha256,
    validate_manifest,
)


def _activate(raw):
    raw["status"] = "frozen"
    raw["correctness"]["calibration_required"] = False
    raw["decision"].update({
        "calibration_required": False,
        "minimum_speedup_percent": 5.0,
        "maximum_per_workload_regression_percent": 3.0,
        "maximum_vram_regression_percent": 2.0,
        "tiers_percent": {"S": 5.0, "M": 10.0, "L": 20.0},
        "merge_enabled": True,
    })
    raw["calibration"] = {
        "sessions": 3,
        "benchmark_contract_sha256": benchmark_contract_sha256(raw),
    }


def test_draft_manifest_defines_hardened_nine_case_contract():
    manifest = load_manifest()
    assert manifest.id == "attention-foundation-v1.1-rtx5070ti"
    assert manifest.status == "draft"
    assert not manifest.is_active
    assert manifest.raw["result_schema_version"] == 2
    assert len(manifest.workloads) == 9
    assert sum(workload.scored for workload in manifest.workloads) == 7
    assert {workload.mode for workload in manifest.workloads} == {"prefill", "decode", "guard"}
    assert manifest.raw["measurement"]["fresh_inputs_per_call"] is True
    assert manifest.raw["measurement"]["validate_timed_outputs"] is True


def test_manifest_hash_is_the_canonical_json_hash():
    manifest = load_manifest()
    assert manifest.sha256 == manifest_sha256(manifest.raw)


def test_manifest_hash_is_independent_of_line_endings_and_formatting(tmp_path):
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    compact = tmp_path / "compact.json"
    windows = tmp_path / "windows.json"
    compact.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")
    windows.write_bytes((json.dumps(raw, indent=4) + "\n").replace("\n", "\r\n").encode("utf-8"))
    assert load_manifest(compact).sha256 == load_manifest(windows).sha256


def test_benchmark_contract_hash_survives_lifecycle_and_threshold_changes():
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    expected = benchmark_contract_sha256(raw)
    raw["status"] = "frozen"
    raw["description"] = "frozen after calibration"
    raw["correctness"]["calibration_required"] = False
    raw["decision"] = {
        "calibration_required": False,
        "minimum_speedup_percent": 7.0,
        "maximum_per_workload_regression_percent": 7.0,
        "maximum_vram_regression_percent": 2.0,
        "tiers_percent": {"S": 7.0, "M": 14.0, "L": 28.0},
        "merge_enabled": True,
    }
    assert benchmark_contract_sha256(raw) == expected


def test_duplicate_workload_id_is_rejected():
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    raw["workloads"][1]["id"] = raw["workloads"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(raw)


def test_hardened_manifest_requires_every_integrity_guarantee():
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    raw["measurement"]["validate_timed_outputs"] = False
    with pytest.raises(ValueError, match="validate_timed_outputs=true"):
        validate_manifest(raw)


def test_frozen_manifest_requires_calibrated_thresholds():
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    raw["status"] = "frozen"
    raw["decision"].update({
        "calibration_required": True,
        "minimum_speedup_percent": None,
        "maximum_per_workload_regression_percent": None,
        "maximum_vram_regression_percent": None,
        "tiers_percent": None,
        "merge_enabled": False,
    })
    with pytest.raises(ValueError, match="calibration"):
        validate_manifest(raw)


def test_frozen_manifest_can_be_active_after_calibration(tmp_path):
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    _activate(raw)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_manifest(path).is_active


def test_frozen_manifest_rejects_mismatched_calibration_contract(tmp_path):
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    _activate(raw)
    raw["calibration"]["benchmark_contract_sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_manifest(path)


def test_frozen_manifest_rejects_non_numeric_threshold():
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    _activate(raw)
    raw["decision"]["minimum_speedup_percent"] = "5"
    with pytest.raises(ValueError, match="finite non-negative"):
        validate_manifest(raw)
