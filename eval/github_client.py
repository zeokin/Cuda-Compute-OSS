"""Thin wrapper over the ``gh`` CLI, shared by eval.pr_bot (the periodic
gate-chain sweep) and eval.copycat_guard's real-time single-PR check (the
sensitive-paths-guard.yml-adjacent workflow) so neither has to duplicate
subprocess/GitHub plumbing.

Every method is one subprocess call, so a fake stand-in is trivial to write
for tests (see eval/tests/test_pr_bot.py's ``FakeClient``) -- this class
itself is not unit-tested, only the pure decision logic that consumes it is.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass


@dataclass
class PRInfo:
    number: int
    title: str
    author: str
    is_draft: bool
    head_sha: str
    body: str = ""
    url: str = ""
    labels: tuple[str, ...] = ()
    updated_at: str = ""
    merge_state_status: str = ""
    mergeable: str = ""


@dataclass
class ReviewInfo:
    reviewer: str
    state: str
    submitted_at: str
    commit_id: str = ""
    author_association: str = ""


_TRANSIENT_MARKERS = (
    "502", "503", "504", "gateway timeout", "timeout",
    "temporarily unavailable", "secondary rate limit", "rate limit",
    "connection reset", "connection refused", "eof", "tls handshake",
    "could not resolve host",
)


def _is_transient(text: str) -> bool:
    """True for gh/GitHub failures worth retrying rather than aborting on:
    5xx gateways, timeouts, secondary rate limits, dropped connections."""
    low = (text or "").lower()
    return any(marker in low for marker in _TRANSIENT_MARKERS)


class GitHubClient:
    def __init__(self, repo: str):
        self.repo = repo

    def _exec(self, args, *, repo: bool = True, check: bool = True,
              retries: int = 3, backoff: float = 2.0) -> subprocess.CompletedProcess:
        """Run one ``gh`` command, retrying transient failures (5xx / timeout /
        secondary rate limit / dropped connection) with exponential backoff
        before giving up. A single transient GitHub hiccup must not abort the
        whole PR sweep (see eval.pr_bot.run_once). On persistent failure this
        raises when ``check`` (matching subprocess semantics), else returns the
        failed result. ``repo=False`` omits the ``-R`` flag, for ``gh api``
        calls that carry the repo in the URL."""
        cmd = ["gh", *args] + (["-R", self.repo] if repo else [])
        last = None
        for attempt in range(max(1, retries)):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                return proc
            last = proc
            transient = _is_transient(proc.stderr) or _is_transient(proc.stdout)
            if not transient or attempt == retries - 1:
                break
            time.sleep(backoff * (2 ** attempt))
        if check and last is not None:
            raise subprocess.CalledProcessError(
                last.returncode, cmd, last.stdout, last.stderr)
        return last

    def _run(self, *args: str) -> str:
        return self._exec(list(args)).stdout

    def list_prs(self, state: str = "open") -> list:
        out = self._run("pr", "list", "--state", state, "-L", "300", "--json",
                        "number,title,author,isDraft,headRefOid,body,url,labels,updatedAt,"
                        "mergeStateStatus,mergeable")
        data = json.loads(out)
        return [
            PRInfo(number=d["number"], title=d["title"],
                   author=d["author"]["login"], is_draft=d["isDraft"],
                   head_sha=d["headRefOid"], body=d.get("body") or "",
                   url=d.get("url") or "",
                   labels=tuple(lbl["name"] for lbl in d.get("labels", [])),
                   updated_at=d.get("updatedAt") or "",
                   merge_state_status=d.get("mergeStateStatus") or "",
                   mergeable=d.get("mergeable") or "")
            for d in data
        ]

    def get_pr(self, pr_number: int) -> PRInfo:
        out = self._run("pr", "view", str(pr_number), "--json",
                        "number,title,author,isDraft,headRefOid,body,url,labels,updatedAt,"
                        "mergeStateStatus,mergeable")
        d = json.loads(out)
        return PRInfo(number=d["number"], title=d["title"],
                      author=d["author"]["login"], is_draft=d["isDraft"],
                      head_sha=d["headRefOid"], body=d.get("body") or "",
                      url=d.get("url") or "",
                      labels=tuple(lbl["name"] for lbl in d.get("labels", [])),
                      updated_at=d.get("updatedAt") or "",
                      merge_state_status=d.get("mergeStateStatus") or "",
                      mergeable=d.get("mergeable") or "")

    def get_diff(self, pr_number: int) -> str:
        return self._run("pr", "diff", str(pr_number))

    def get_comments(self, pr_number: int) -> list:
        out = self._run("pr", "view", str(pr_number), "--json", "comments")
        data = json.loads(out)
        return [c["body"] for c in data.get("comments", [])]

    def get_commit_messages(self, pr_number: int) -> str:
        out = self._run("pr", "view", str(pr_number), "--json", "commits")
        data = json.loads(out)
        parts = []
        for commit in data.get("commits", []):
            headline = commit.get("messageHeadline") or ""
            body = commit.get("messageBody") or ""
            parts.append(f"{headline}\n{body}".strip())
        return "\n\n".join(part for part in parts if part)

    def get_reviews(self, pr_number: int) -> list[ReviewInfo]:
        out = self._exec(
            ["api", f"repos/{self.repo}/pulls/{pr_number}/reviews"],
            repo=False,
        ).stdout
        data = json.loads(out)
        return [
            ReviewInfo(
                reviewer=(d.get("user") or {}).get("login") or "",
                state=d.get("state") or "",
                submitted_at=d.get("submitted_at") or "",
                commit_id=d.get("commit_id") or "",
                author_association=d.get("author_association") or "",
            )
            for d in data
        ]

    def post_comment(self, pr_number: int, body: str) -> None:
        self._exec(["pr", "comment", str(pr_number), "--body", body])

    def add_label(self, pr_number: int, label: str) -> None:
        self._exec(["pr", "edit", str(pr_number), "--add-label", label])

    def remove_label(self, pr_number: int, label: str) -> None:
        self._exec(["pr", "edit", str(pr_number), "--remove-label", label],
                   check=False)

    def close_pr(self, pr_number: int, reason: str) -> None:
        self._exec(["pr", "close", str(pr_number), "--comment", reason])
