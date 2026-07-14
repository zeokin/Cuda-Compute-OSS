"""Tests for eval.result_bot -- mock GPU JSON to ledger/dashboard, no GitHub."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.gpu_batch import EvalSpec, QueueItem, mock_result
from eval.result_bot import (
    already_recorded,
    best_transform,
    comment_body,
    load_result,
    process_results,
    result_entry,
    track_from_config,
)


def _write_payload(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mock_payload(pr=11, fill="random"):
    return mock_result(
        QueueItem(pr=pr, title=f"PR {pr}", author="alice", head_sha=f"sha{pr}", url=f"https://x/{pr}"),
        EvalSpec(transforms="mine", fill=fill),
    )


def test_track_from_config():
    assert track_from_config({"fill": "random"}) == "full-rank"
    assert track_from_config({"fill": "lowrank"}) == "low-rank"
    assert track_from_config({"fill": "decaying-spectrum"}) == "decaying-spectrum"


def test_result_entry_first_admission_is_baseline():
    payload = _mock_payload()
    entry = result_entry(payload, [])
    assert entry["verdict"] == "BASELINE"
    assert entry["track"] == "full-rank"
    assert entry["transform"] == "mine"
    assert entry["mock"] is True


def test_best_transform_reads_eval_payload():
    name, result = best_transform(_mock_payload())
    assert name == "mine"
    assert result["improvement"] is True


def test_already_recorded_uses_pr_and_commit():
    entry = {"pr": 1, "commit": "abc"}
    assert already_recorded([entry], {"pr": 1, "commit": "abc"})
    assert not already_recorded([entry], {"pr": 1, "commit": "def"})


def test_process_results_is_idempotent_and_writes_dashboard():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        result_path = _write_payload(root / "result.json", _mock_payload(pr=12))
        ledger = root / "ledger.jsonl"
        dashboard = root / "results.json"

        first = process_results([result_path], ledger_path=ledger, dashboard_results=dashboard)
        second = process_results([result_path], ledger_path=ledger, dashboard_results=dashboard)

        assert first[0]["verdict"] == "BASELINE"
        assert second[0]["verdict"] == "BASELINE"
        assert len(ledger.read_text().strip().splitlines()) == 1
        data = json.loads(dashboard.read_text())
        assert data["status"]["gpu"] == "RTX 5090"
        assert data["prs"][0]["label"] == "BASELINE"


def test_comment_body_contains_marker_and_verdict():
    entry = result_entry(_mock_payload(pr=13), [])
    body = comment_body(entry)
    assert "<!-- cco-result:13:sha13 -->" in body
    assert "eval:BASELINE" in body


def test_load_result_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = _write_payload(Path(d) / "result.json", _mock_payload(pr=14))
        assert load_result(path)["pr"] == 14


def test_candidate_transform_scores_declared_not_best():
    from eval.result_bot import candidate_transform
    # 'rsvd' is the best-scoring transform, but the PR declared 'nystrom' ->
    # the PR must be scored on nystrom, not credited for rsvd's result.
    payload = {"transform": "nystrom", "eval": {"best": "rsvd", "transforms": {
        "rsvd": {"score": 13.0}, "nystrom": {"score": 17.0}}}}
    name, res = candidate_transform(payload)
    assert name == "nystrom" and res["score"] == 17.0
    # nothing declared -> fall back to the best transform
    p2 = {"eval": {"best": "rsvd", "transforms": {"rsvd": {"score": 13.0}}}}
    assert candidate_transform(p2)[0] == "rsvd"


def test_result_entry_uses_declared_track_and_transform():
    payload = _mock_payload(pr=21, fill="lowrank")
    payload["track"] = "low-rank"
    payload["transform"] = "mine"
    entry = result_entry(payload, [])
    assert entry["track"] == "low-rank" and entry["transform"] == "mine"


def test_process_results_skips_blocked_state():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rp = _write_payload(root / "r.json", {
            "pr": 5, "head_sha": "s5", "state": "needs_rebase",
            "detail": "conflicts with main"})
        ledger = root / "ledger.jsonl"
        out = process_results([rp], ledger_path=ledger,
                              dashboard_results=root / "results.json")
        assert out[0]["state"] == "needs_rebase"
        # a blocked PR gets NO ledger entry (no verdict)
        assert not ledger.exists() or ledger.read_text().strip() == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
