"""Append-only attention verdict ledger and pure leaderboard projection."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ledger(path: str | Path) -> list[dict]:
    ledger = Path(path)
    if not ledger.exists():
        return []
    return [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def decision_entry(decision: dict) -> dict:
    main_path = decision.get("main_artifact")
    candidate_path = decision.get("candidate_artifact")
    return {
        "benchmark": decision["benchmark"],
        "manifest_sha256": decision["manifest_sha256"],
        "pr": int(decision["pr"]),
        "title": decision.get("title", ""),
        "author": decision.get("author", ""),
        "url": decision.get("url", ""),
        "main_commit": decision.get("main_commit"),
        "candidate_commit": decision.get("candidate_commit"),
        "evaluated_candidate_commit": decision.get("evaluated_candidate_commit"),
        "verdict": decision["verdict"],
        "label": decision.get("label"),
        "tracks": decision.get("tracks", {}),
        "reasons": decision.get("reasons", []),
        "main_artifact_sha256": file_sha256(main_path) if main_path else None,
        "candidate_artifact_sha256": file_sha256(candidate_path) if candidate_path else None,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def append_entry(path: str | Path, entry: dict) -> bool:
    ledger = Path(path)
    existing = read_ledger(ledger)
    key = (entry.get("benchmark"), entry.get("pr"), entry.get("candidate_commit"))
    for old in existing:
        old_key = (old.get("benchmark"), old.get("pr"), old.get("candidate_commit"))
        if old_key == key:
            comparable_old = dict(old)
            comparable_new = dict(entry)
            comparable_old.pop("recorded_at", None)
            comparable_new.pop("recorded_at", None)
            if comparable_old != comparable_new:
                raise ValueError("attention ledger key already exists with different content")
            return False
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return True


def build_projection(entries: list[dict]) -> dict:
    by_benchmark: dict[str, list[dict]] = {}
    for entry in entries:
        by_benchmark.setdefault(entry["benchmark"], []).append(entry)
    benchmarks = {}
    for benchmark, cells in by_benchmark.items():
        admitted = [cell for cell in cells if cell.get("verdict") == "ADMIT"]
        frontier = admitted[-1] if admitted else None
        benchmarks[benchmark] = {
            "evaluations": len(cells),
            "admitted": len(admitted),
            "frontier": frontier,
            "history": cells,
        }
    return {
        "schema_version": 1,
        "updated": max(
            (entry.get("recorded_at", "") for entry in entries),
            default="",
        ),
        "benchmarks": benchmarks,
    }


def write_projection(path: str | Path, entries: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_projection(entries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "append_entry", "build_projection", "decision_entry", "file_sha256",
    "read_ledger", "write_projection",
]
