from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from eval.attention_batch import (
    _merge_current_main,
    _process_decision,
    ensure_processing_allowed,
    load_queue,
    select_batch,
)
from eval.attention_manifest import load_manifest


def test_queue_only_selects_current_attention_benchmark(tmp_path):
    manifest = load_manifest()
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"queue": [
        {"pr": 1, "head_sha": "a", "benchmark": manifest.id, "state": "attention_eval_pending", "position": 2},
        {"pr": 2, "head_sha": "b", "benchmark": "legacy", "state": "eval_pending", "position": 1},
        {"pr": 3, "head_sha": "c", "benchmark": manifest.id, "state": "blocked", "position": 1},
    ]}), encoding="utf-8")
    items = load_queue(path, manifest)
    assert [item.pr for item in items] == [1]


def test_select_batch_zero_means_all():
    manifest = load_manifest()
    assert select_batch([], 0) == []


def test_draft_manifest_cannot_process_prs():
    manifest = load_manifest()
    with pytest.raises(RuntimeError, match="cannot process"):
        ensure_processing_allowed(manifest, manifest.id)


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo, name, content, message):
    (repo / name).write_text(content, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)


def test_candidate_is_locally_merged_with_new_main(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _commit(repo, "base.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature")
    _commit(repo, "feature.txt", "feature\n", "feature")
    feature_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    _commit(repo, "main.txt", "main\n", "main")
    main_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "feature")

    evaluated_sha = _merge_current_main(repo, main_sha)

    assert evaluated_sha not in {None, feature_sha, main_sha}
    assert _git(repo, "rev-parse", f"{evaluated_sha}^1") == feature_sha
    assert _git(repo, "rev-parse", f"{evaluated_sha}^2") == main_sha
    assert not _git(repo, "status", "--porcelain")


def test_conflicting_candidate_is_blocked_and_merge_is_aborted(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _commit(repo, "shared.txt", "base\n", "base")
    _git(repo, "checkout", "-b", "feature")
    _commit(repo, "shared.txt", "feature\n", "feature")
    feature_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    _commit(repo, "shared.txt", "main\n", "main")
    main_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "feature")

    assert _merge_current_main(repo, main_sha) is None
    assert _git(repo, "rev-parse", "HEAD") == feature_sha
    assert not _git(repo, "status", "--porcelain")


def test_processing_refuses_a_stale_pr_head_before_mutation(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="new-head\n")

    monkeypatch.setattr("eval.attention_batch._run", fake_run)
    with pytest.raises(RuntimeError, match="head changed"):
        _process_decision(
            {
                "pr": 9,
                "candidate_commit": "old-head",
                "main_commit": "main",
                "verdict": "REJECT",
                "merge": False,
            },
            repo="owner/repo",
            python=["python"],
            workdir=tmp_path,
            state_branch="state",
        )
    assert len(calls) == 1
    assert calls[0][:3] == ["gh", "pr", "view"]


def test_admit_checks_state_and_pins_head_before_merge(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["gh", "pr", "view"]:
            return SimpleNamespace(stdout="head\n")
        if args[:2] == ["gh", "api"]:
            return SimpleNamespace(stdout="main\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("eval.attention_batch._run", fake_run)
    monkeypatch.setattr("eval.attention_batch._clone", lambda *args, **kwargs: None)
    monkeypatch.setattr("eval.attention_batch._preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr("eval.attention_batch._remove", lambda *args, **kwargs: None)
    monkeypatch.setattr("eval.attention_batch._publish_decision", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "eval.attention_batch.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
    )
    _process_decision(
        {
            "pr": 9,
            "candidate_commit": "head",
            "main_commit": "main",
            "verdict": "ADMIT",
            "label": "eval:S",
            "merge": True,
            "reasons": ["gain"],
        },
        repo="owner/repo",
        python=["python"],
        workdir=tmp_path,
        state_branch="state",
    )
    checks_index = next(i for i, call in enumerate(calls) if call[:3] == ["gh", "pr", "checks"])
    comment_index = next(i for i, call in enumerate(calls) if call[:3] == ["gh", "pr", "comment"])
    assert checks_index < comment_index
    merge = next(call for call in calls if call[:3] == ["gh", "pr", "merge"])
    assert merge[-2:] == ["--match-head-commit", "head"]
