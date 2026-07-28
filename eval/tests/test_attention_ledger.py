from __future__ import annotations

import json

import pytest

from eval.attention_ledger import (
    append_entry,
    build_projection,
    decision_entry,
    file_sha256,
    read_ledger,
)


def _entry(pr=1, verdict="ADMIT", commit="abc"):
    return {
        "benchmark": "bench",
        "pr": pr,
        "candidate_commit": commit,
        "verdict": verdict,
        "recorded_at": "2026-01-01T00:00:00+00:00",
    }


def test_append_is_idempotent_for_same_decision(tmp_path):
    path = tmp_path / "ledger.jsonl"
    assert append_entry(path, _entry())
    assert not append_entry(path, _entry())
    assert read_ledger(path) == [_entry()]


def test_append_rejects_rewriting_existing_key(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_entry(path, _entry())
    with pytest.raises(ValueError, match="different content"):
        append_entry(path, _entry(verdict="REJECT"))


def test_projection_uses_latest_admitted_entry_as_frontier():
    entries = [_entry(pr=1), _entry(pr=2, verdict="NONE", commit="def"), _entry(pr=3, commit="ghi")]
    result = build_projection(entries)["benchmarks"]["bench"]
    assert result["evaluations"] == 3
    assert result["admitted"] == 2
    assert result["frontier"]["pr"] == 3


def test_decision_entry_records_source_and_evaluated_commits(tmp_path):
    main = tmp_path / "main.json"
    candidate = tmp_path / "candidate.json"
    main.write_text("{}\n", encoding="utf-8")
    candidate.write_text("{}\n", encoding="utf-8")
    entry = decision_entry({
        "benchmark": "bench",
        "manifest_sha256": "manifest",
        "pr": 4,
        "main_commit": "main",
        "candidate_commit": "source-head",
        "evaluated_candidate_commit": "source-head-plus-main",
        "verdict": "ADMIT",
        "main_artifact": str(main),
        "candidate_artifact": str(candidate),
    })
    assert entry["candidate_commit"] == "source-head"
    assert entry["evaluated_candidate_commit"] == "source-head-plus-main"
    assert entry["main_artifact_sha256"] == file_sha256(main)
    assert entry["candidate_artifact_sha256"] == file_sha256(candidate)
