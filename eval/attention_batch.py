"""Sequential manual GPU runner for the attention-foundation phase.

The safe default prints a plan.  ``--run`` evaluates but does not mutate
GitHub.  ``--process`` is additionally required to label/close/merge, and is
hard-locked while the manifest is not frozen and active.
"""
from __future__ import annotations

import argparse
import json
import secrets
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .attention_benchmark import merge_artifact_shards
from .attention_decision import compare_artifacts
from .attention_ledger import append_entry, decision_entry, read_ledger, write_projection
from .attention_manifest import DEFAULT_MANIFEST, AttentionManifest, load_manifest


DEFAULT_QUEUE = "dashboard/data.json"
DEFAULT_WORKDIR = "_attention_batch_work"
DEFAULT_RESULTS = "attention-results"
EVAL_LABELS = ("eval:BASELINE", "eval:S", "eval:M", "eval:L", "eval:none", "eval:REJECT", "eval:attention-draft")


@dataclass(frozen=True)
class AttentionQueueItem:
    pr: int
    title: str
    author: str
    head_sha: str
    benchmark: str
    position: int | None = None
    url: str = ""


def load_queue(path: str | Path, manifest: AttentionManifest) -> list[AttentionQueueItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for entry in raw.get("queue", []):
        if entry.get("benchmark") != manifest.id:
            continue
        if entry.get("state") not in {"attention_eval_pending", "eval_pending"}:
            continue
        items.append(AttentionQueueItem(
            pr=int(entry["pr"]),
            title=entry.get("title", ""),
            author=entry.get("author", ""),
            head_sha=entry.get("head_sha", ""),
            benchmark=entry["benchmark"],
            position=entry.get("position"),
            url=entry.get("url", ""),
        ))
    return sorted(items, key=lambda item: (item.position or item.pr, item.pr))


def select_batch(items: list[AttentionQueueItem], limit: int) -> list[AttentionQueueItem]:
    return items if limit <= 0 else items[:limit]


def ensure_processing_allowed(manifest: AttentionManifest, confirmation: str | None) -> None:
    if not manifest.is_active:
        raise RuntimeError(
            f"benchmark {manifest.id} is {manifest.status!r} and cannot process PRs"
        )
    if manifest.path != DEFAULT_MANIFEST.resolve():
        raise RuntimeError(
            "only the repository default manifest may process PRs; "
            "historical or custom manifests are evaluation-only"
        )
    if confirmation != manifest.id:
        raise RuntimeError(
            "processing requires --confirm-benchmark with the exact active benchmark id"
        )


def _retry_readonly(func, path, exc_info) -> None:
    exc = exc_info[1]
    if not isinstance(exc, PermissionError):
        raise exc
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def _remove(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_retry_readonly)


def _run(args: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )


def _python(active_python: bool) -> list[str]:
    return [sys.executable] if active_python else ["uv", "run", "python"]


def plan_item(item: AttentionQueueItem, *, repo: str, workdir: Path, results: Path) -> list[str]:
    main = workdir / f"pr-{item.pr}-main"
    candidate = workdir / f"pr-{item.pr}-candidate"
    return [
        f"clone current main into {main}",
        f"checkout PR #{item.pr} at {item.head_sha} into {candidate}",
        "merge current main into the local candidate tree; block conflicts",
        "run compile, full CPU/GPU tests, and attention correctness preflight",
        "run balanced shards: main -> PR -> PR -> main",
        f"write immutable artifacts and decision under {results}",
        "process only if the manifest is frozen, calibrated, and merge-enabled",
    ]


def _benchmark_command(
    python: list[str], *, output: Path, shard_index: int, manifest_path: str, seed: int
) -> list[str]:
    return [
        *python,
        "-m", "eval.attention_benchmark",
        "--manifest", manifest_path,
        "--official",
        "--shard-index", str(shard_index),
        "--shard-count", "2",
        "--seed", str(seed),
        "--output", str(output),
    ]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clone(repo: str, destination: Path) -> None:
    _run(["gh", "repo", "clone", repo, str(destination)])


def _blocked_decision(
    item: AttentionQueueItem,
    manifest: AttentionManifest,
    reason: str,
) -> dict:
    return {
        "schema_version": 1,
        "benchmark": manifest.id,
        "manifest_sha256": manifest.sha256,
        "benchmark_contract_sha256": manifest.benchmark_contract_sha256,
        "pr": item.pr,
        "candidate_commit": item.head_sha,
        "verdict": "BLOCKED",
        "label": None,
        "merge": False,
        "reasons": [reason],
    }


def _merge_current_main(checkout: Path, main_sha: str) -> str | None:
    """Create the exact candidate tree to evaluate without changing the PR.

    A preceding winner may have changed ``main`` after this PR was opened.  A
    local merge lets the batch continue against the new frontier.  Conflicts
    are returned as a blocked result and must be resolved by the contributor.
    """
    _run(["git", "config", "user.name", "Cuda-Compute-OSS attention evaluator"], cwd=checkout)
    _run(["git", "config", "user.email", "attention-evaluator@users.noreply.github.com"], cwd=checkout)
    merged = subprocess.run(
        ["git", "merge", "--no-edit", main_sha],
        cwd=checkout,
        text=True,
        capture_output=True,
    )
    if merged.returncode != 0:
        subprocess.run(["git", "merge", "--abort"], cwd=checkout, capture_output=True)
        return None
    return _run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).stdout.strip()


def _preflight(checkout: Path, python: list[str]) -> None:
    _run([*python, "-m", "compileall", "-q", "matmul", "strategy", "eval", "attention", "tests", "examples"], cwd=checkout)
    _run([*python, "-m", "pytest", "tests/", "strategy/tests/", "eval/tests/", "-q"], cwd=checkout)
    _run([*python, "-m", "strategy.smoke"], cwd=checkout)


def run_item(
    item: AttentionQueueItem,
    *,
    manifest: AttentionManifest,
    repo: str,
    workdir: Path,
    results_dir: Path,
    clean: bool,
    active_python: bool,
) -> Path:
    main_checkout = workdir / f"pr-{item.pr}-main"
    candidate_checkout = workdir / f"pr-{item.pr}-candidate"
    item_dir = results_dir.resolve() / f"pr-{item.pr}-{item.head_sha[:12]}"
    decision_path = item_dir / "decision.json"
    for path in (main_checkout, candidate_checkout):
        if path.exists():
            if not clean:
                raise FileExistsError(f"{path} exists; pass --clean to replace it")
            _remove(path)
    if item_dir.exists() and clean:
        _remove(item_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    item_dir.mkdir(parents=True, exist_ok=True)

    _clone(repo, main_checkout)
    _run(["git", "checkout", "--detach", "origin/main"], cwd=main_checkout)
    main_sha = _run(["git", "rev-parse", "HEAD"], cwd=main_checkout, capture=True).stdout.strip()
    _clone(repo, candidate_checkout)
    _run(["gh", "pr", "checkout", str(item.pr)], cwd=candidate_checkout)
    actual_sha = _run(["git", "rev-parse", "HEAD"], cwd=candidate_checkout, capture=True).stdout.strip()
    if actual_sha != item.head_sha:
        raise RuntimeError(f"PR #{item.pr} head changed: queued {item.head_sha}, checked out {actual_sha}")
    _run(["git", "fetch", "origin", "main"], cwd=candidate_checkout)
    fetched_main = _run(
        ["git", "rev-parse", "origin/main"], cwd=candidate_checkout, capture=True
    ).stdout.strip()
    if fetched_main != main_sha:
        _write(decision_path, _blocked_decision(
            item, manifest, "main changed while preparing evaluation; retry the batch"
        ))
        return decision_path
    evaluated_sha = _merge_current_main(candidate_checkout, main_sha)
    if evaluated_sha is None:
        _write(decision_path, _blocked_decision(
            item, manifest, "PR conflicts with current main; resolve conflicts and requeue"
        ))
        return decision_path

    python = _python(active_python)
    _preflight(candidate_checkout, python)
    manifest_relative = f"benchmarks/{manifest.path.name}"
    evaluation_seed = secrets.randbits(63)
    main_shards = [item_dir / "main-0.json", item_dir / "main-1.json"]
    pr_shards = [item_dir / "candidate-0.json", item_dir / "candidate-1.json"]
    # Coarse balanced interleaving reduces monotonic clock/temperature drift.
    sequence = (
        (main_checkout, main_shards[0], 0),
        (candidate_checkout, pr_shards[0], 0),
        (candidate_checkout, pr_shards[1], 1),
        (main_checkout, main_shards[1], 1),
    )
    for checkout, output, shard_index in sequence:
        _run(
            _benchmark_command(
                python,
                output=output,
                shard_index=shard_index,
                manifest_path=manifest_relative,
                seed=evaluation_seed,
            ),
            cwd=checkout,
        )
    main_artifact = merge_artifact_shards([_read(path) for path in main_shards])
    candidate_artifact = merge_artifact_shards([_read(path) for path in pr_shards])
    main_path = item_dir / "main.json"
    candidate_path = item_dir / "candidate.json"
    _write(main_path, main_artifact)
    _write(candidate_path, candidate_artifact)
    decision = compare_artifacts(
        main_artifact,
        candidate_artifact,
        manifest,
        expected_candidate_commit=evaluated_sha,
    )
    decision["evaluated_candidate_commit"] = decision.get("candidate_commit")
    decision["candidate_commit"] = item.head_sha
    decision.update({
        "pr": item.pr,
        "title": item.title,
        "author": item.author,
        "url": item.url,
        "main_artifact": str(main_path),
        "candidate_artifact": str(candidate_path),
    })
    _write(decision_path, decision)
    return decision_path


def _process_decision(
    decision: dict,
    *,
    repo: str,
    python: list[str],
    workdir: Path,
    state_branch: str,
) -> None:
    pr = int(decision["pr"])
    current_head = _run(
        ["gh", "pr", "view", str(pr), "-R", repo, "--json", "headRefOid", "--jq", ".headRefOid"],
        capture=True,
    ).stdout.strip()
    if current_head != decision.get("candidate_commit"):
        raise RuntimeError(f"PR #{pr} head changed after evaluation; refusing stale decision")
    current_main = _run(
        ["gh", "api", f"repos/{repo}/commits/main", "--jq", ".sha"],
        capture=True,
    ).stdout.strip()
    if current_main != decision.get("main_commit"):
        raise RuntimeError("main changed after evaluation; refusing stale decision")
    if decision["verdict"] == "ADMIT" and decision.get("merge"):
        _run(["gh", "pr", "checks", str(pr), "-R", repo, "--required"])

    marker = f"<!-- cco-attention-result:{pr}:{decision.get('candidate_commit')} -->"
    summary = marker + "\n" + "Attention foundation verdict: **" + decision["verdict"] + "**\n\n" + "\n".join(
        f"- {reason}" for reason in decision.get("reasons", [])
    )
    for label in EVAL_LABELS:
        subprocess.run(["gh", "pr", "edit", str(pr), "-R", repo, "--remove-label", label], capture_output=True)
    if decision.get("label"):
        _run(["gh", "pr", "edit", str(pr), "-R", repo, "--add-label", decision["label"]])
    _run(["gh", "pr", "comment", str(pr), "-R", repo, "--body", summary])

    if decision["verdict"] in {"REJECT", "NONE"}:
        _run(["gh", "pr", "close", str(pr), "-R", repo])
    elif decision["verdict"] != "ADMIT" or not decision.get("merge"):
        raise RuntimeError(f"refusing to process non-terminal verdict {decision['verdict']}")
    else:
        _run([
            "gh", "pr", "merge", str(pr), "-R", repo, "--squash", "--delete-branch",
            "--match-head-commit", str(decision["candidate_commit"]),
        ])

        postmerge = workdir / f"postmerge-{pr}"
        _remove(postmerge)
        _clone(repo, postmerge)
        _run(["git", "checkout", "--detach", "origin/main"], cwd=postmerge)
        _preflight(postmerge, python)
        _remove(postmerge)
    _publish_decision(
        decision,
        repo=repo,
        workdir=workdir,
        state_branch=state_branch,
    )


def _copy_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != data:
            raise RuntimeError(f"refusing to rewrite immutable artifact {destination}")
        return
    destination.write_bytes(data)


def _publish_decision(
    decision: dict,
    *,
    repo: str,
    workdir: Path,
    state_branch: str,
) -> None:
    state = workdir / "attention-state-publish"
    _remove(state)
    _run([
        "gh", "repo", "clone", repo, str(state), "--",
        "--branch", state_branch, "--single-branch",
    ])
    artifact_dir = (
        state / "dashboard" / "attention-artifacts" /
        f"pr-{decision['pr']}-{str(decision['candidate_commit'])[:12]}"
    )
    _copy_immutable(Path(decision["main_artifact"]), artifact_dir / "main.json")
    _copy_immutable(Path(decision["candidate_artifact"]), artifact_dir / "candidate.json")
    published_decision = dict(decision)
    published_decision["main_artifact"] = "main.json"
    published_decision["candidate_artifact"] = "candidate.json"
    published_decision_path = artifact_dir / "decision.json"
    if published_decision_path.exists():
        existing = json.loads(published_decision_path.read_text(encoding="utf-8"))
        if existing != published_decision:
            raise RuntimeError(f"refusing to rewrite immutable artifact {published_decision_path}")
    else:
        _write(published_decision_path, published_decision)

    ledger_path = state / "eval" / "attention-ledger.jsonl"
    append_entry(ledger_path, decision_entry(decision))
    write_projection(
        state / "dashboard" / "attention-results.json",
        read_ledger(ledger_path),
    )
    _run(["git", "config", "user.name", "Cuda-Compute-OSS attention evaluator"], cwd=state)
    _run(["git", "config", "user.email", "attention-evaluator@users.noreply.github.com"], cwd=state)
    _run(["git", "add", "eval/attention-ledger.jsonl", "dashboard/attention-results.json", "dashboard/attention-artifacts"], cwd=state)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=state).returncode != 0
    if changed:
        _run(["git", "commit", "-m", f"dashboard: record attention PR #{decision['pr']}"], cwd=state)
        _run(["git", "push", "origin", f"HEAD:{state_branch}"], cwd=state)
    _remove(state)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.attention_batch",
        description="Preview, evaluate, and optionally process the attention PR queue.",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--repo", default="zeokin/Cuda-Compute-OSS")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--process", action="store_true")
    parser.add_argument("--confirm-benchmark")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--active-python", action="store_true")
    parser.add_argument("--state-branch", default="bot/dashboard-state")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.process:
            if not args.run:
                raise RuntimeError("--process requires --run")
            ensure_processing_allowed(manifest, args.confirm_benchmark)
        batch = select_batch(load_queue(args.queue, manifest), args.limit)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not batch:
        print(f"No queued PRs for {manifest.id}.")
        return 0

    workdir = Path(args.workdir)
    results_dir = Path(args.results_dir)
    python = _python(args.active_python)
    for item in batch:
        print(f"PR #{item.pr} ({item.author}): {item.title}")
        if not args.run:
            for step in plan_item(item, repo=args.repo, workdir=workdir, results=results_dir):
                print(f"  {step}")
            continue
        try:
            decision_path = run_item(
                item,
                manifest=manifest,
                repo=args.repo,
                workdir=workdir,
                results_dir=results_dir,
                clean=args.clean,
                active_python=args.active_python,
            )
            print(f"  wrote {decision_path}")
            if args.process:
                _process_decision(
                    _read(decision_path),
                    repo=args.repo,
                    python=python,
                    workdir=workdir,
                    state_branch=args.state_branch,
                )
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"error: PR #{item.pr}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
