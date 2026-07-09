"""Append-only PR evaluation ledger + the dashboard/data.json projection.

The ledger (``ledger.jsonl``) is the single source of truth: one JSON line
per sealed evaluation, in landing order, never edited or reordered after the
fact (docs/whitepaper.md Sec 6.3). ``dashboard/data.json`` is a pure,
disposable PROJECTION of the ledger -- always fully rebuilt from it, never
hand-patched, so the two can never drift out of sync.

Pure stdlib (json/pathlib) -- no GPU, no network, testable anywhere.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Verdicts that count as an admitted improvement on a track's frontier.
ADMITTED_VERDICTS = frozenset({"BASELINE", "S", "M", "L"})


def append_entry(ledger_path: str | os.PathLike, entry: dict) -> None:
    """Append one evaluation record as a single JSON line.

    Expected keys: date, pr, track, transform, verdict, score, accuracy,
    latency_s, peak_vram_bytes, peak_vram_mib, flop_ratio, seed, commit,
    title, url. Extra keys are passed through unchanged -- callers may attach
    provenance freely. Never rewrites or reorders prior lines.
    """
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def read_ledger(ledger_path: str | os.PathLike) -> list[dict]:
    """Read every entry, in landing order. Missing file -> empty ledger."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def is_append_only(old_lines: list[str], new_lines: list[str]) -> bool:
    """True iff ``new_lines`` is ``old_lines`` with zero or more lines
    appended -- never edited, reordered, or removed. Intended for a CI check
    (docs/sn74-emission-strategy.md's ledger-integrity requirement) comparing
    the ledger before/after a PR that touches it."""
    return new_lines[: len(old_lines)] == old_lines


def frontier_score(entries: list[dict], track: str) -> float:
    """Current best admitted ``score`` on ``track`` (0.0 if none admitted)."""
    admitted = [e["score"] for e in entries
                if e.get("track") == track and e.get("verdict") in ADMITTED_VERDICTS]
    return max(admitted, default=0.0)


def reference_anchor(entries: list[dict], track: str) -> float:
    """The first-ever admitted score on ``track`` -- the fixed anchor
    eval.label sizes tiers against (0.0 if nothing admitted yet)."""
    for e in entries:
        if e.get("track") == track and e.get("verdict") in ADMITTED_VERDICTS:
            return e["score"]
    return 0.0


def build_dashboard_data(
    entries: list[dict],
    *,
    gpu: str,
    accuracy_floors: dict[str, float],
    roadmap: list[dict],
    updated: str,
) -> dict:
    """Rebuild the full dashboard/data.json payload from the ledger.

    Always a full rebuild, never an incremental patch, so the dashboard can
    never drift from what the ledger actually says.
    """
    tracks_seen = sorted({e["track"] for e in entries if "track" in e} | set(accuracy_floors))

    tracks: dict = {}
    status_tracks: dict = {}
    for track in tracks_seen:
        landed = [
            {"name": e.get("transform", ""), "pr": e.get("pr"), "date": e.get("date"),
             "score": e.get("score"), "label": e.get("verdict")}
            for e in entries
            if e.get("track") == track and e.get("verdict") in ADMITTED_VERDICTS
        ]
        tracks[track] = {"landed": landed}
        status_tracks[track] = {
            "frontier_score": frontier_score(entries, track),
            "accuracy_floor": accuracy_floors.get(track, 0.8),
        }

    prs = [
        {
            "num": e.get("pr"), "title": e.get("title", ""), "track": e.get("track"),
            "label": e.get("verdict"), "accuracy": e.get("accuracy"),
            "latency_s": e.get("latency_s"), "peak_vram_mib": e.get("peak_vram_mib"),
            "flop_ratio": e.get("flop_ratio"), "url": e.get("url", ""),
        }
        for e in entries
    ]

    return {
        "updated": updated,
        "status": {"gpu": gpu, "tracks": status_tracks},
        "tracks": tracks,
        "prs": prs,
        "roadmap": roadmap,
    }


def write_dashboard_data(dashboard_path: str | os.PathLike, data: dict) -> None:
    path = Path(dashboard_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
