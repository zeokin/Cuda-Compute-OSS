from __future__ import annotations

from eval.attention_benchmark import _summary, environment_mismatches, merge_artifact_shards
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
        "measurement": {"warmups": 2, "repetitions": 2},
        "workloads": [{
            "id": "w",
            "correctness": {"passed": True},
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
    second["workloads"][0]["candidate"]["timing"] = _summary([3.0, 4.0])
    second["workloads"][0]["candidate"]["peak_incremental_vram_bytes"] = 30
    merged = merge_artifact_shards([base, second])
    assert merged["measurement"]["repetitions"] == 4
    assert merged["workloads"][0]["candidate"]["timing"]["samples_ms"] == [1.0, 2.0, 3.0, 4.0]
    assert merged["workloads"][0]["candidate"]["peak_incremental_vram_bytes"] == 30
