from __future__ import annotations

from eval.attention_benchmark import (
    _summary,
    _timed_validation_summary,
    environment_mismatches,
    merge_artifact_shards,
)
from eval.attention_manifest import load_manifest


def test_timing_summary_preserves_samples_and_reports_p90():
    result = _summary([5.0, 1.0, 4.0, 2.0, 3.0])
    assert result["samples_ms"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["median_ms"] == 3.0
    assert result["p90_ms"] == 5.0


def test_environment_match_requires_the_pinned_era():
    manifest = load_manifest()
    environment = {
        "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
        "os": "Windows",
        "python": "3.12.12",
        "torch": "2.11.0+cu128",
        "tf32_matmul": False,
        "deterministic_algorithms": False,
        "cudnn_benchmark": False,
    }
    assert environment_mismatches(manifest, environment) == []
    environment["gpu_name"] = "NVIDIA GeForce RTX 5090"
    assert "gpu_name" in environment_mismatches(manifest, environment)[0]


def test_timed_validation_rejects_any_bad_measured_output():
    qualities = [
        {"relative_frobenius_error": 0.0, "maximum_absolute_error": 0.0, "finite": True},
        {"relative_frobenius_error": 0.01, "maximum_absolute_error": 0.01, "finite": True},
    ]
    result = _timed_validation_summary(qualities, {
        "max_relative_frobenius_error": 0.005,
        "max_absolute_error": 0.05,
    })
    assert not result["passed"]
    assert result["repetitions"] == 2


def test_merge_artifact_shards_combines_raw_samples_and_peak():
    base = {
        "schema_version": 1,
        "benchmark": "bench",
        "manifest_sha256": "a" * 64,
        "benchmark_contract_sha256": "b" * 64,
        "candidate": "module:callable",
        "commit": "abc",
        "dirty": False,
        "seed": 1,
        "environment": {"gpu": "same"},
        "measurement": {
            "warmups": 2,
            "repetitions": 2,
            "seed_policy": "os-random-per-call-published-after",
            "fresh_inputs_per_call": True,
            "validate_timed_outputs": True,
            "candidate_import_after_reference": True,
            "runtime_state_guard": True,
        },
        "workloads": [{
            "id": "w",
            "input_seeds": {
                "correctness": 1,
                "production": {"warmups": [2, 3], "measured": [4, 5]},
                "candidate": {"warmups": [6, 7], "measured": [8, 9]},
            },
            "correctness": {
                "passed": True,
                "timed_output_validation": {
                    "passed": True,
                    "repetitions": 2,
                    "maximum_relative_frobenius_error": 0.0,
                    "maximum_absolute_error": 0.0,
                    "finite": True,
                },
            },
            "production": {
                "timing": _summary([2.0, 3.0]),
                "peak_incremental_vram_bytes": 10,
            },
            "candidate": {
                "timing": _summary([1.0, 2.0]),
                "peak_incremental_vram_bytes": 20,
            },
        }],
    }
    import copy

    second = copy.deepcopy(base)
    second["workloads"][0]["input_seeds"] = {
        "correctness": 10,
        "production": {"warmups": [11, 12], "measured": [13, 14]},
        "candidate": {"warmups": [15, 16], "measured": [17, 18]},
    }
    second["workloads"][0]["candidate"]["timing"] = _summary([3.0, 4.0])
    second["workloads"][0]["candidate"]["peak_incremental_vram_bytes"] = 30
    merged = merge_artifact_shards([base, second])
    assert merged["measurement"]["repetitions"] == 4
    assert merged["measurement"]["fresh_inputs_per_call"] is True
    assert merged["measurement"]["seed_policy"] == "os-random-per-call-published-after"
    assert merged["workloads"][0]["correctness"]["timed_output_validation"]["repetitions"] == 4
    assert merged["workloads"][0]["input_seeds"]["candidate"]["measured"] == [8, 9, 17, 18]
    assert merged["workloads"][0]["candidate"]["timing"]["samples_ms"] == [1.0, 2.0, 3.0, 4.0]
    assert merged["workloads"][0]["candidate"]["peak_incremental_vram_bytes"] == 30
