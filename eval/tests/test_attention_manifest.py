from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eval.attention_manifest import DEFAULT_MANIFEST, load_manifest, validate_manifest


def test_draft_manifest_defines_nine_workloads_and_seven_scored_cases():
    manifest = load_manifest()
    assert manifest.id == "attention-foundation-v1-rtx5070ti"
    assert manifest.status == "draft"
    assert not manifest.is_active
    assert len(manifest.workloads) == 9
    assert sum(workload.scored for workload in manifest.workloads) == 7
    assert {workload.mode for workload in manifest.workloads} == {"prefill", "decode", "guard"}


def test_manifest_hash_is_the_file_hash():
    import hashlib

    manifest = load_manifest()
    assert manifest.sha256 == hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()


def test_duplicate_workload_id_is_rejected():
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    raw["workloads"][1]["id"] = raw["workloads"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(raw)


def test_frozen_manifest_requires_calibrated_thresholds():
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    raw["status"] = "frozen"
    with pytest.raises(ValueError, match="calibration"):
        validate_manifest(raw)


def test_frozen_manifest_can_be_active_after_calibration(tmp_path):
    raw = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    raw["status"] = "frozen"
    raw["decision"].update({
        "calibration_required": False,
        "minimum_speedup_percent": 5.0,
        "maximum_per_workload_regression_percent": 3.0,
        "maximum_vram_regression_percent": 2.0,
        "tiers_percent": {"S": 5.0, "M": 10.0, "L": 20.0},
        "merge_enabled": True,
    })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_manifest(path).is_active
