"""Phase 3 result processor: GPU JSON -> labels, ledger, dashboard data.

Default mode is dry-run. Passing ``--write`` updates GitHub labels/comments.
Ledger and dashboard writes are local file writes because they are intended to
be committed by a maintainer workflow after review.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import label as verdict_label
from .github_client import GitHubClient
from .ledger import (
    append_entry,
    build_dashboard_data,
    frontier_score,
    read_ledger,
    reference_anchor,
    write_dashboard_data,
)
from .pr_bot import GPU_QUEUE_LABEL

DEFAULT_LEDGER = "eval/ledger.jsonl"
DEFAULT_DASHBOARD_RESULTS = "dashboard/results.json"
RESULT_MARKER = "<!-- cco-result:{pr}:{commit} -->"
EVAL_LABELS = {
    "eval:BASELINE", "eval:S", "eval:M", "eval:L",
    "eval:none", "eval:REJECT",
}


def load_result(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def track_from_config(config: dict) -> str:
    fill = config.get("fill", "random")
    if fill == "random":
        return "full-rank"
    if fill == "lowrank":
        return "low-rank"
    return str(fill)


def best_transform(payload: dict) -> tuple[str, dict]:
    ev = payload["eval"]
    name = ev.get("best") or (ev.get("ranking") or [None])[0]
    if not name:
        raise ValueError("result has no best transform")
    return name, ev["transforms"][name]


def entry_key(entry: dict) -> tuple:
    return entry.get("pr"), entry.get("commit")


def already_recorded(entries: list[dict], entry: dict) -> bool:
    key = entry_key(entry)
    return any(entry_key(existing) == key for existing in entries)


def find_recorded(entries: list[dict], pr: int | None, commit: str | None) -> dict | None:
    for existing in entries:
        if existing.get("pr") == pr and existing.get("commit") == commit:
            return existing
    return None


def result_entry(payload: dict, entries: list[dict]) -> dict:
    ev = payload["eval"]
    config = ev.get("config", {})
    transform, result = best_transform(payload)
    track = track_from_config(config)
    verdict = verdict_label.label(
        result,
        frontier_score(entries, track),
        reference_anchor(entries, track),
    )
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "date": now,
        "pr": payload.get("pr"),
        "title": payload.get("title", ""),
        "author": payload.get("author", ""),
        "url": payload.get("url", ""),
        "commit": payload.get("head_sha", ""),
        "track": track,
        "transform": transform,
        "verdict": verdict["verdict"],
        "verdict_reason": verdict.get("reason"),
        "delta_pct": verdict.get("delta_pct"),
        "score": result.get("score", 0.0),
        "accuracy": result.get("accuracy"),
        "latency_s": result.get("latency_s"),
        "peak_vram_bytes": result.get("peak_vram_bytes"),
        "peak_vram_mib": result.get("peak_vram_mib"),
        "flop_ratio": result.get("flop_ratio_vs_exact"),
        "seed": config.get("seed"),
        "gpu": config.get("device", "RTX 5090"),
        "n": config.get("n"),
        "pairs": config.get("pairs"),
        "dtype": config.get("dtype"),
        "fill": config.get("fill"),
        "rank_m": config.get("rank_m"),
        "mock": bool(payload.get("mock")),
    }


def comment_body(entry: dict) -> str:
    marker = RESULT_MARKER.format(pr=entry["pr"], commit=entry["commit"])
    reason = f"\nReason: {entry['verdict_reason']}" if entry.get("verdict_reason") else ""
    return (
        f"{marker}\n"
        f"GPU evaluation complete on {entry.get('gpu', 'RTX 5090')}.\n\n"
        f"- Verdict: `eval:{entry['verdict']}`{reason}\n"
        f"- Track: `{entry['track']}` / transform `{entry['transform']}`\n"
        f"- Accuracy: `{entry.get('accuracy')}`\n"
        f"- Latency: `{entry.get('latency_s')}` seconds\n"
        f"- Peak VRAM: `{entry.get('peak_vram_mib')}` MiB\n"
        f"- FLOP ratio vs exact: `{entry.get('flop_ratio')}`\n"
        f"- Score: `{entry.get('score')}`\n"
        f"- Seed: `{entry.get('seed')}`\n"
    )


def apply_github_result(
    client: GitHubClient,
    entry: dict,
    *,
    close_rejected: bool = False,
) -> None:
    pr = int(entry["pr"])
    label = f"eval:{entry['verdict']}"
    for old in EVAL_LABELS:
        if old != label:
            client.remove_label(pr, old)
    client.remove_label(pr, GPU_QUEUE_LABEL)
    client.add_label(pr, label)
    marker = RESULT_MARKER.format(pr=entry["pr"], commit=entry["commit"])
    if not any(marker in body for body in client.get_comments(pr)):
        client.post_comment(pr, comment_body(entry))
    if close_rejected and entry["verdict"] == "REJECT":
        client.close_pr(pr, "Closed after GPU evaluation: eval:REJECT")


def process_results(
    paths: list[str | Path],
    *,
    ledger_path: str | Path = DEFAULT_LEDGER,
    dashboard_results: str | Path = DEFAULT_DASHBOARD_RESULTS,
    write_github: bool = False,
    repo: str = "zeokin/Cuda-Compute-OSS",
    close_rejected: bool = False,
) -> list[dict]:
    entries = read_ledger(ledger_path)
    client = GitHubClient(repo) if write_github else None
    processed = []
    for path in paths:
        payload = load_result(path)
        existing = find_recorded(entries, payload.get("pr"), payload.get("head_sha", ""))
        if existing is not None:
            entry = existing
        else:
            entry = result_entry(payload, entries)
            append_entry(ledger_path, entry)
            entries.append(entry)
        if client is not None:
            apply_github_result(client, entry, close_rejected=close_rejected)
        processed.append(entry)

    data = build_dashboard_data(
        entries,
        gpu="RTX 5090",
        accuracy_floors={"full-rank": 0.8, "low-rank": 0.8, "decaying-spectrum": 0.8},
        roadmap=[
            {"phase": 1, "target": "governance and CPU validation", "status": "done"},
            {"phase": 2, "target": "queued sequential GPU batches", "status": "ready"},
            {"phase": 3, "target": "automated result labels and ledger", "status": "in progress"},
            {"phase": 4, "target": "public dashboard", "status": "in progress"},
        ],
        updated=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    write_dashboard_data(dashboard_results, data)
    return processed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.result_bot",
        description="Process GPU result JSON into eval labels, ledger, and dashboard data.",
    )
    parser.add_argument("results", nargs="+")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--dashboard-results", default=DEFAULT_DASHBOARD_RESULTS)
    parser.add_argument("--repo", default="zeokin/Cuda-Compute-OSS")
    parser.add_argument("--write", action="store_true",
                        help="write labels/comments to GitHub. Default is dry-run.")
    parser.add_argument("--close-rejected", action="store_true",
                        help="with --write, close PRs that receive eval:REJECT")
    args = parser.parse_args(argv)

    entries = process_results(
        args.results,
        ledger_path=args.ledger,
        dashboard_results=args.dashboard_results,
        write_github=args.write,
        repo=args.repo,
        close_rejected=args.close_rejected,
    )
    for entry in entries:
        mode = "mock " if entry.get("mock") else ""
        print(f"PR #{entry['pr']}: {mode}eval:{entry['verdict']} score={entry['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
