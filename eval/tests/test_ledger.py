"""Tests for eval.ledger -- pure stdlib, no GPU.

    python eval/tests/test_ledger.py        (or)   python -m pytest eval/tests -q
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.ledger import (
    append_entry, read_ledger, is_append_only, frontier_score,
    reference_anchor, build_dashboard_data, write_dashboard_data,
)


def _entry(pr, track, verdict, score, **extra):
    e = {"date": "2026-07-06", "pr": pr, "track": track, "transform": "rsvd",
         "verdict": verdict, "score": score, "accuracy": 0.99, "latency_s": 0.1,
         "peak_vram_bytes": 1e9, "peak_vram_mib": 953.7, "flop_ratio": 2.0,
         "seed": 42, "commit": "abc123", "title": f"PR {pr}", "url": f"https://x/{pr}"}
    e.update(extra)
    return e


def test_append_and_read_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ledger.jsonl"
        append_entry(path, _entry(1, "full-rank", "BASELINE", 1.0))
        append_entry(path, _entry(2, "full-rank", "S", 1.03))
        entries = read_ledger(path)
        assert len(entries) == 2
        assert entries[0]["pr"] == 1 and entries[1]["pr"] == 2


def test_read_missing_ledger_is_empty():
    assert read_ledger("/tmp/definitely-does-not-exist-cco-ledger.jsonl") == []


def test_is_append_only():
    old = ["a", "b"]
    assert is_append_only(old, ["a", "b", "c"])
    assert is_append_only(old, ["a", "b"])
    assert not is_append_only(old, ["a", "x"])       # edited a line
    assert not is_append_only(old, ["b", "a", "c"])  # reordered
    assert not is_append_only(old, ["a"])            # shrank


def test_frontier_and_reference_anchor():
    entries = [
        _entry(1, "full-rank", "BASELINE", 1.0),
        _entry(2, "full-rank", "none", 5.0),   # not admitted -- must not count
        _entry(3, "full-rank", "S", 1.05),
        _entry(4, "low-rank", "BASELINE", 2.0),
    ]
    assert frontier_score(entries, "full-rank") == 1.05
    assert reference_anchor(entries, "full-rank") == 1.0  # first admitted, not best
    assert frontier_score(entries, "low-rank") == 2.0
    assert frontier_score(entries, "no-such-track") == 0.0


def test_dashboard_rebuild_is_pure_projection_of_ledger():
    entries = [
        _entry(1, "full-rank", "BASELINE", 1.0),
        _entry(2, "full-rank", "REJECT", 0.0),
    ]
    data = build_dashboard_data(
        entries, gpu="RTX 5070 Ti",
        accuracy_floors={"full-rank": 0.8, "low-rank": 0.95},
        roadmap=[{"phase": 0, "target": "matmul arena", "status": "live"}],
        updated="2026-07-06",
    )
    assert data["updated"] == "2026-07-06"
    assert data["status"]["gpu"] == "RTX 5070 Ti"
    assert data["status"]["tracks"]["full-rank"]["frontier_score"] == 1.0
    assert data["status"]["tracks"]["low-rank"]["accuracy_floor"] == 0.95
    assert len(data["tracks"]["full-rank"]["landed"]) == 1   # REJECT excluded
    assert len(data["prs"]) == 2                             # every PR still listed
    assert data["roadmap"][0]["phase"] == 0


def test_write_dashboard_data_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.json"
        write_dashboard_data(path, {"updated": "x", "prs": []})
        assert json.loads(path.read_text())["updated"] == "x"


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
