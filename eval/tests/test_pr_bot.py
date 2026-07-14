"""Tests for eval.pr_bot's gate chain -- pure decision logic, no GitHub I/O.

process_pr() takes already-fetched data and performs no I/O itself, so every
test here uses plain fixtures; run_once()/FakeClient below exercise the
orchestration layer with a fake in-memory GitHub, never real `gh` calls.

    python eval/tests/test_pr_bot.py        (or)   python -m pytest eval/tests -q
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval import copycat_guard
from eval.pr_bot import (
    CHANGES_REQUESTED_LABEL,
    GPU_QUEUE_LABEL,
    MAX_OPEN_PRS_PER_AUTHOR,
    READY_NON_GPU_LABEL,
    PRInfo,
    already_evaluated,
    already_queued,
    build_queue_dashboard,
    changed_files,
    excess_open_prs,
    has_coding_agent_coauthor,
    has_scorecard,
    latest_maintainer_change_request,
    merge_conflict_comment_time,
    process_pr,
    run_once,
    protected_paths,
    write_queue_dashboard,
    ReviewInfo,
)

# Matches .github/workflows/labeler.yml's status:needs-scorecard detector
# (an actual filled-in accuracy/latency number or RESULT_JSON, not just a
# checked checkbox on an empty template).
SCORECARD_BODY = """\
## Summary
Adds a new transform.

| metric | value |
|---|---|
| accuracy | 0.991 |
| latency | 0.42s |
"""

NO_SCORECARD_BODY = "Adds a new transform. No testing done yet."
CODING_AGENT_COMMIT = """\
fix: useful change

Co-authored-by: Cursor <cursoragent@cursor.com>
"""
HUMAN_COAUTHOR_COMMIT = """\
fix: useful change

Co-authored-by: Alice Example <alice@example.com>
"""

SOME_DIFF = """\
diff --git a/strategy/transforms.py b/strategy/transforms.py
--- a/strategy/transforms.py
+++ b/strategy/transforms.py
@@ -10,0 +11,3 @@
+class Foo(Transform):
+    name = "foo"
+    def basis(self, n, m, backend, dtype, A=None, B=None): return None
"""

PROTECTED_DIFF = """\
diff --git a/eval/evaluator.py b/eval/evaluator.py
--- a/eval/evaluator.py
+++ b/eval/evaluator.py
@@ -1,0 +2,1 @@
+print("changed scoring")
"""


def _pr(
    number=1,
    author="alice",
    is_draft=False,
    head_sha="sha1",
    body=SCORECARD_BODY,
    updated_at="2026-07-10T00:00:00Z",
    merge_state_status="CLEAN",
    mergeable="MERGEABLE",
):
    return PRInfo(number=number, title=f"feat: PR {number}", author=author,
                 is_draft=is_draft, head_sha=head_sha, body=body,
                 updated_at=updated_at,
                 merge_state_status=merge_state_status,
                 mergeable=mergeable)


def test_draft_is_skipped():
    out = process_pr(_pr(is_draft=True), SOME_DIFF, [], frozenset(), [])
    assert out.action == "skip_draft"


def test_blocked_contributor_is_closed():
    out = process_pr(_pr(author="badactor"), SOME_DIFF, [], frozenset({"badactor"}), [])
    assert out.action == "close_blocked"


def test_process_pr_closes_excess_open_pr():
    pr = _pr(number=1, author="alice")
    out = process_pr(pr, SOME_DIFF, [], frozenset(), [], frozenset({1}))
    assert out.action == "close_excess_open_pr"
    assert str(MAX_OPEN_PRS_PER_AUTHOR) in out.detail


def test_has_coding_agent_coauthor_detects_agent_footers_only():
    assert has_coding_agent_coauthor(CODING_AGENT_COMMIT)
    assert has_coding_agent_coauthor("Co-authored-by: Claude <noreply@anthropic.com>")
    assert has_coding_agent_coauthor("Co-authored-by: Codex <codex@openai.com>")
    assert not has_coding_agent_coauthor(HUMAN_COAUTHOR_COMMIT)
    assert not has_coding_agent_coauthor("")


def test_coding_agent_coauthor_blocks_pr_routing():
    pr = _pr(number=1, body=SCORECARD_BODY)
    out = process_pr(
        pr,
        SOME_DIFF,
        [],
        frozenset(),
        [],
        commit_messages=CODING_AGENT_COMMIT,
    )
    assert out.action == "close_coding_agent_coauthor"
    assert out.label is None


def test_merge_conflict_without_prior_comment_requests_resolution():
    pr = _pr(number=1, merge_state_status="DIRTY", mergeable="CONFLICTING")
    out = process_pr(pr, SOME_DIFF, [], frozenset(), [])
    assert out.action == "needs_merge_conflict_resolution"


def test_merge_conflict_closes_after_grace_window():
    pr = _pr(number=1, head_sha="sha1", merge_state_status="DIRTY", mergeable="CONFLICTING")
    warned_at = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    marker = f"<!-- cco-merge-conflict:sha1:{warned_at.isoformat()} -->"
    out = process_pr(
        pr,
        SOME_DIFF,
        [marker],
        frozenset(),
        [],
        now=warned_at + timedelta(hours=12, minutes=1),
    )
    assert out.action == "close_stale_merge_conflict"


def test_maintainer_hold_label_prevents_stale_merge_conflict_close():
    pr = _pr(number=1, head_sha="sha1", merge_state_status="DIRTY", mergeable="CONFLICTING")
    pr.labels = ("status:maintainer-review",)
    warned_at = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    marker = f"<!-- cco-merge-conflict:sha1:{warned_at.isoformat()} -->"
    out = process_pr(
        pr,
        SOME_DIFF,
        [marker],
        frozenset(),
        [],
        now=warned_at + timedelta(hours=12, minutes=1),
    )
    assert out.action == "needs_merge_conflict_resolution"


def test_changed_files_from_diff():
    assert changed_files(SOME_DIFF) == frozenset({"strategy/transforms.py"})


def test_protected_paths_are_detected():
    assert protected_paths(PROTECTED_DIFF) == ("eval/evaluator.py",)


def test_protected_path_is_closed_immediately():
    out = process_pr(_pr(author="alice"), PROTECTED_DIFF, [], frozenset(), [])
    assert out.action == "close_protected_path"
    assert "maintainer-owned files" in out.detail


def test_unknown_pr_kind_is_flagged():
    pr = PRInfo(number=1, title="PR 1", author="alice", is_draft=False, head_sha="sha1",
                body="Just some changes.")
    out = process_pr(pr, "diff --git a/matmul/x.py b/matmul/x.py\n+++ b/matmul/x.py\n+pass\n",
                     [], frozenset(), [])
    assert out.action == "close_missing_pr_kind"


def test_fix_pr_does_not_require_gpu_scorecard():
    pr = PRInfo(number=1, title="fix: PR 1", author="alice", is_draft=False, head_sha="sha1",
                body=NO_SCORECARD_BODY)
    out = process_pr(pr, SOME_DIFF, [], frozenset(), [])
    assert out.action == "non_gpu_review"
    assert out.label == READY_NON_GPU_LABEL


def test_copycat_block_beats_scorecard_check():
    # Even with a perfect scorecard, an exact copy of an earlier PR is blocked
    # before the scorecard is ever consulted.
    orig_fp = copycat_guard.fingerprint(SOME_DIFF)
    out = process_pr(_pr(author="mallory"), SOME_DIFF, [], frozenset(),
                     [("original-author", orig_fp)])
    assert out.action == "copycat_block"


def test_copycat_check_excludes_own_earlier_prs():
    # The same author reusing their OWN earlier diff is not copying.
    orig_fp = copycat_guard.fingerprint(SOME_DIFF)
    out = process_pr(_pr(author="alice", body=SCORECARD_BODY), SOME_DIFF, [], frozenset(),
                     [("alice", orig_fp)])
    assert out.action not in {"copycat_block", "copycat_warn"}


def test_queue_marker_keeps_pr_in_eval_pending_state():
    marker = "<!-- cco-eval:sha1 -->"
    out = process_pr(_pr(head_sha="sha1"), SOME_DIFF, [marker], frozenset(), [])
    assert out.action == "eval_pending"


def test_missing_scorecard_is_flagged():
    out = process_pr(_pr(body=NO_SCORECARD_BODY), SOME_DIFF, [], frozenset(), [])
    assert out.action == "close_missing_scorecard"


def test_clean_pr_with_no_runner_is_eval_pending():
    out = process_pr(_pr(body=SCORECARD_BODY), SOME_DIFF, [], frozenset(), [], run_eval=None)
    assert out.action == "eval_pending"
    assert out.label == GPU_QUEUE_LABEL


def test_clean_pr_with_runner_is_evaluated():
    out = process_pr(_pr(body=SCORECARD_BODY), SOME_DIFF, [], frozenset(), [],
                     run_eval=lambda pr: {"track": "full-rank", "verdict": "S"})
    assert out.action == "evaluated"
    assert "S" in out.detail


def test_has_scorecard_matches_labeler_ymls_detector():
    assert has_scorecard(SCORECARD_BODY)
    assert has_scorecard("here is my RESULT_JSON {...}")
    assert has_scorecard("| latency | exact 632.06 ms |")
    assert has_scorecard("| accuracy | 1.0 (reference) |")
    assert not has_scorecard(NO_SCORECARD_BODY)
    assert not has_scorecard("")
    assert not has_scorecard(None)


def test_already_evaluated_helper():
    assert already_evaluated(1, ["x", "<!-- cco-result:1:abc -->"])
    assert not already_evaluated(1, ["x", "<!-- cco-eval:abc -->"])


def test_already_queued_helper():
    assert already_queued(["hello", "<!-- cco-eval:abc -->", "world"], "abc")
    assert not already_queued(["hello", "world"], "abc")


def test_merge_conflict_comment_time_ignores_other_heads_and_picks_latest():
    comments = [
        "<!-- cco-merge-conflict:sha0:2026-07-10T01:00:00+00:00 -->",
        "<!-- cco-merge-conflict:sha1:2026-07-10T02:00:00+00:00 -->",
        "<!-- cco-merge-conflict:sha1:2026-07-10T03:00:00+00:00 -->",
    ]
    when = merge_conflict_comment_time(comments, "sha1")
    assert when == datetime(2026, 7, 10, 3, 0, tzinfo=timezone.utc)


def _review(
    reviewer="maintainer",
    state="CHANGES_REQUESTED",
    submitted_at="2026-07-10T08:00:00Z",
    commit_id="sha1",
    author_association="MEMBER",
):
    return ReviewInfo(
        reviewer=reviewer,
        state=state,
        submitted_at=submitted_at,
        commit_id=commit_id,
        author_association=author_association,
    )


def test_latest_maintainer_change_request_uses_latest_review_per_reviewer():
    reviews = [
        _review(reviewer="alice", state="CHANGES_REQUESTED", submitted_at="2026-07-10T01:00:00Z"),
        _review(reviewer="alice", state="APPROVED", submitted_at="2026-07-10T02:00:00Z"),
        _review(reviewer="bob", state="CHANGES_REQUESTED", submitted_at="2026-07-10T03:00:00Z"),
    ]
    review = latest_maintainer_change_request(reviews, "sha1")
    assert review.reviewer == "bob"


def test_maintainer_change_request_blocks_current_head():
    requested_at = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    out = process_pr(
        _pr(number=1, head_sha="sha1"),
        SOME_DIFF,
        [],
        frozenset(),
        [],
        now=requested_at + timedelta(hours=1),
        reviews=[_review(submitted_at=requested_at.isoformat(), commit_id="sha1")],
    )
    assert out.action == "maintainer_changes_requested"
    assert out.label == CHANGES_REQUESTED_LABEL


def test_stale_maintainer_change_request_closes_after_12_hours():
    requested_at = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    out = process_pr(
        _pr(number=1, head_sha="sha1"),
        SOME_DIFF,
        [],
        frozenset(),
        [],
        now=requested_at + timedelta(hours=12, minutes=1),
        reviews=[_review(submitted_at=requested_at.isoformat(), commit_id="sha1")],
    )
    assert out.action == "close_stale_maintainer_changes"


def test_maintainer_change_request_resets_after_new_head_sha():
    requested_at = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    out = process_pr(
        _pr(number=1, head_sha="sha2"),
        SOME_DIFF,
        [],
        frozenset(),
        [],
        now=requested_at + timedelta(hours=12, minutes=1),
        reviews=[_review(submitted_at=requested_at.isoformat(), commit_id="sha1")],
    )
    assert out.action == "eval_pending"


def test_maintainer_hold_label_prevents_stale_review_close():
    requested_at = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    pr = _pr(number=1, head_sha="sha1")
    pr.labels = ("status:maintainer-review",)
    out = process_pr(
        pr,
        SOME_DIFF,
        [],
        frozenset(),
        [],
        now=requested_at + timedelta(hours=12, minutes=1),
        reviews=[_review(submitted_at=requested_at.isoformat(), commit_id="sha1")],
    )
    assert out.action == "maintainer_changes_requested"


def test_non_maintainer_change_request_is_ignored():
    requested_at = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    out = process_pr(
        _pr(number=1, head_sha="sha1"),
        SOME_DIFF,
        [],
        frozenset(),
        [],
        now=requested_at + timedelta(hours=12, minutes=1),
        reviews=[
            _review(
                submitted_at=requested_at.isoformat(),
                commit_id="sha1",
                author_association="CONTRIBUTOR",
            )
        ],
    )
    assert out.action == "eval_pending"


def test_excess_open_prs_keeps_two_oldest_prs_per_author():
    pr1 = _pr(number=1, author="alice", updated_at="2026-07-10T08:00:00Z")
    pr2 = _pr(number=2, author="alice", updated_at="2026-07-10T09:00:00Z")
    pr3 = _pr(number=3, author="alice", updated_at="2026-07-10T10:00:00Z")
    pr4 = _pr(number=4, author="bob", updated_at="2026-07-10T07:00:00Z")
    assert excess_open_prs([pr1, pr2, pr3, pr4]) == frozenset({3})


def test_excess_open_prs_ignores_recent_updates_and_closes_newer_prs():
    pr1 = _pr(number=1, author="alice", updated_at="2026-07-10T11:00:00Z")
    pr2 = _pr(number=2, author="alice", updated_at="2026-07-10T09:00:00Z")
    pr3 = _pr(number=3, author="alice", updated_at="2026-07-10T10:00:00Z")
    assert excess_open_prs([pr1, pr2, pr3]) == frozenset({3})


class FakeClient:
    """In-memory stand-in for GitHubClient -- no subprocess/network calls."""

    def __init__(self, prs, diffs, comments=None, commit_messages=None, reviews=None):
        self._prs = prs               # dict: state -> list[PRInfo]
        self._diffs = diffs           # dict: pr_number -> diff text
        self._comments = comments or {}
        self._commit_messages = commit_messages or {}
        self._reviews = reviews or {}
        self.actions = []             # records of what WOULD have been written

    def list_prs(self, state="open"):
        if state == "all":
            return self._prs.get("all", [])
        return self._prs.get(state, [])

    def get_diff(self, pr_number):
        return self._diffs.get(pr_number, "")

    def get_comments(self, pr_number):
        return self._comments.get(pr_number, [])

    def get_commit_messages(self, pr_number):
        return self._commit_messages.get(pr_number, "")

    def get_reviews(self, pr_number):
        return self._reviews.get(pr_number, [])

    def post_comment(self, pr_number, body):
        self.actions.append(("post_comment", pr_number, body))

    def add_label(self, pr_number, label):
        self.actions.append(("add_label", pr_number, label))

    def remove_label(self, pr_number, label):
        self.actions.append(("remove_label", pr_number, label))

    def close_pr(self, pr_number, reason):
        self.actions.append(("close_pr", pr_number, reason))


def test_run_once_dry_run_never_writes():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY)
    pr2 = _pr(number=2, author="badactor", body=SCORECARD_BODY)
    client = FakeClient(
        prs={"all": [pr1, pr2], "open": [pr1, pr2]},
        diffs={1: SOME_DIFF, 2: SOME_DIFF + "\n+# a harmless extra line"},
    )
    blocked_path_written = False
    outcomes = run_once(client, dry_run=True)
    assert len(outcomes) == 2
    assert client.actions == [], "dry_run must never call any write method"


def test_run_once_live_mode_applies_actions():
    pr1 = _pr(number=1, author="badactor", body=SCORECARD_BODY)
    client = FakeClient(prs={"all": [pr1], "open": [pr1]}, diffs={1: SOME_DIFF})
    # No blocked-contributors.txt in this sandbox -> not blocked; force the
    # missing-scorecard close path instead by using a body with no scorecard,
    # so we can observe a real write action deterministically without touching
    # the filesystem-backed blocked list.
    pr1.body = NO_SCORECARD_BODY
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "close_missing_scorecard"
    assert ("close_pr", 1, outcomes[0].detail) in client.actions


def test_run_once_live_mode_closes_stale_maintainer_change_request():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY, head_sha="sha1")
    stale = (datetime.now(timezone.utc) - timedelta(hours=12, minutes=5)).replace(microsecond=0)
    client = FakeClient(
        prs={"all": [pr1], "open": [pr1]},
        diffs={1: SOME_DIFF},
        reviews={1: [_review(submitted_at=stale.isoformat(), commit_id="sha1")]},
    )
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "close_stale_maintainer_changes"
    assert ("close_pr", 1, outcomes[0].detail) in client.actions


def test_run_once_live_mode_labels_active_maintainer_change_request():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY, head_sha="sha1")
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
    client = FakeClient(
        prs={"all": [pr1], "open": [pr1]},
        diffs={1: SOME_DIFF},
        reviews={1: [_review(submitted_at=recent.isoformat(), commit_id="sha1")]},
    )
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "maintainer_changes_requested"
    assert ("add_label", 1, CHANGES_REQUESTED_LABEL) in client.actions
    assert ("remove_label", 1, GPU_QUEUE_LABEL) in client.actions


def test_run_once_live_mode_blocks_coding_agent_coauthor_footer():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY)
    client = FakeClient(
        prs={"all": [pr1], "open": [pr1]},
        diffs={1: SOME_DIFF},
        commit_messages={1: CODING_AGENT_COMMIT},
    )
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "close_coding_agent_coauthor"
    assert ("close_pr", 1, outcomes[0].detail) in client.actions


def test_run_once_live_mode_labels_gpu_queue():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY)
    client = FakeClient(prs={"all": [pr1], "open": [pr1]}, diffs={1: SOME_DIFF})
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "eval_pending"
    assert ("add_label", 1, GPU_QUEUE_LABEL) in client.actions
    assert any("next batched GPU evaluation" in a[2] for a in client.actions
               if a[0] == "post_comment")
    assert ("remove_label", 1, "status:needs-scorecard") in client.actions


def test_run_once_live_mode_labels_fix_pr_non_gpu_ready():
    pr1 = PRInfo(number=1, title="fix: PR 1", author="alice", is_draft=False, head_sha="sha1",
                 body=NO_SCORECARD_BODY)
    client = FakeClient(prs={"all": [pr1], "open": [pr1]}, diffs={1: SOME_DIFF})
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "non_gpu_review"
    assert ("add_label", 1, READY_NON_GPU_LABEL) in client.actions
    assert ("remove_label", 1, GPU_QUEUE_LABEL) in client.actions
    assert not any(a[0] == "post_comment" for a in client.actions)


def test_run_once_live_mode_does_not_repeat_queue_comment():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY, head_sha="sha1")
    client = FakeClient(
        prs={"all": [pr1], "open": [pr1]},
        diffs={1: SOME_DIFF},
        comments={1: ["<!-- cco-eval:sha1 -->"]},
    )
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "eval_pending"
    assert ("add_label", 1, GPU_QUEUE_LABEL) in client.actions
    assert not any(a[0] == "post_comment" for a in client.actions)


def test_run_once_live_mode_comments_once_for_merge_conflict():
    pr1 = _pr(number=1, head_sha="sha1", merge_state_status="DIRTY", mergeable="CONFLICTING")
    client = FakeClient(prs={"all": [pr1], "open": [pr1]}, diffs={1: SOME_DIFF})
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "needs_merge_conflict_resolution"
    comments = [a for a in client.actions if a[0] == "post_comment"]
    assert len(comments) == 1
    assert "resolve them within 12 hours" in comments[0][2]


def test_run_once_live_mode_does_not_repeat_merge_conflict_comment_for_same_head():
    pr1 = _pr(number=1, head_sha="sha1", merge_state_status="DIRTY", mergeable="CONFLICTING")
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
    marker = f"<!-- cco-merge-conflict:sha1:{recent.isoformat()} -->"
    client = FakeClient(
        prs={"all": [pr1], "open": [pr1]},
        diffs={1: SOME_DIFF},
        comments={1: [marker]},
    )
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "needs_merge_conflict_resolution"
    assert not any(a[0] == "post_comment" for a in client.actions)


def test_run_once_live_mode_closes_stale_merge_conflict_after_12_hours():
    pr1 = _pr(number=1, head_sha="sha1", merge_state_status="DIRTY", mergeable="CONFLICTING")
    stale = (datetime.now(timezone.utc) - timedelta(hours=12, minutes=5)).replace(microsecond=0)
    marker = f"<!-- cco-merge-conflict:sha1:{stale.isoformat()} -->"
    client = FakeClient(
        prs={"all": [pr1], "open": [pr1]},
        diffs={1: SOME_DIFF},
        comments={1: [marker]},
    )
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "close_stale_merge_conflict"
    assert ("close_pr", 1, outcomes[0].detail) in client.actions


def test_run_once_live_mode_clears_queue_label_once_already_evaluated():
    pr1 = _pr(number=1, author="alice", head_sha="sha1", body=SCORECARD_BODY)
    client = FakeClient(
        prs={"all": [pr1], "open": [pr1]},
        diffs={1: SOME_DIFF},
        comments={1: ["<!-- cco-result:1:sha1 -->"]},
    )
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "already_evaluated"
    assert ("remove_label", 1, GPU_QUEUE_LABEL) in client.actions


def test_queue_dashboard_only_lists_eval_pending_prs():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY)
    pr2 = _pr(number=2, author="bob", body=SCORECARD_BODY)
    outcomes = [
        process_pr(pr1, SOME_DIFF, ["<!-- cco-result:1:sha1 -->"], frozenset(), []),
        process_pr(pr2, SOME_DIFF, [], frozenset(), []),
    ]
    data = build_queue_dashboard([pr1, pr2], outcomes)
    assert [row["pr"] for row in data["queue"]] == [2]


def test_run_once_live_mode_closes_protected_path():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY)
    client = FakeClient(prs={"all": [pr1], "open": [pr1]}, diffs={1: PROTECTED_DIFF})
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "close_protected_path"
    assert ("close_pr", 1, outcomes[0].detail) in client.actions


def test_run_once_live_mode_closes_older_prs_when_author_exceeds_cap():
    pr1 = _pr(number=1, author="alice", updated_at="2026-07-10T08:00:00Z")
    pr2 = _pr(number=2, author="alice", updated_at="2026-07-10T09:00:00Z")
    pr3 = _pr(number=3, author="alice", updated_at="2026-07-10T10:00:00Z")
    client = FakeClient(
        prs={"all": [pr1, pr2, pr3], "open": [pr1, pr2, pr3]},
        diffs={1: SOME_DIFF, 2: SOME_DIFF, 3: SOME_DIFF},
    )
    outcomes = run_once(client, dry_run=False)
    by_pr = {out.pr: out for out in outcomes}
    assert by_pr[1].action == "eval_pending"
    assert by_pr[2].action == "eval_pending"
    assert by_pr[3].action == "close_excess_open_pr"
    assert ("close_pr", 3, by_pr[3].detail) in client.actions
    assert ("add_label", 1, GPU_QUEUE_LABEL) in client.actions
    assert ("add_label", 2, GPU_QUEUE_LABEL) in client.actions


def test_run_once_originals_include_prs_between_two_open_ones():
    # A merged PR #2 sits between open PRs #1 and #3; #3 copying #2 must be
    # caught even though #2 itself is never "open".
    orig_fp_diff = SOME_DIFF
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY)
    pr2_merged = _pr(number=2, author="bob", body=SCORECARD_BODY)
    pr3 = _pr(number=3, author="mallory", body=SCORECARD_BODY)
    client = FakeClient(
        prs={"all": [pr1, pr2_merged, pr3], "open": [pr1, pr3]},
        diffs={1: "diff --git a/x b/x\n+++ b/x\n+unrelated line one\n",
               2: orig_fp_diff,
               3: orig_fp_diff},
    )
    outcomes = run_once(client, dry_run=True)
    pr3_outcome = next(o for o in outcomes if o.pr == 3)
    assert pr3_outcome.action == "copycat_block", pr3_outcome


def test_queue_dashboard_orders_ready_prs_by_oldest_update_first():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY,
              updated_at="2026-07-10T08:00:00Z")
    pr2 = PRInfo(number=2, title="fix: PR 2", author="bob", is_draft=False, head_sha="sha1",
                 body=NO_SCORECARD_BODY, updated_at="2026-07-10T07:00:00Z")
    pr3 = _pr(number=3, author="carol", body=SCORECARD_BODY,
              updated_at="2026-07-10T09:00:00Z")
    outcomes = [
        process_pr(pr1, SOME_DIFF, [], frozenset(), []),
        process_pr(pr2, SOME_DIFF, [], frozenset(), []),
        process_pr(pr3, SOME_DIFF, [], frozenset(), []),
    ]
    data = build_queue_dashboard([pr1, pr2, pr3], outcomes)
    assert [item["pr"] for item in data["queue"]] == [1, 3]
    assert [item["position"] for item in data["queue"]] == [1, 2]
    assert len(data["open_prs"]) == 3


def test_queue_dashboard_resorts_when_pr_is_updated():
    pr1 = _pr(number=1, updated_at="2026-07-10T08:00:00Z")
    pr2 = _pr(number=2, updated_at="2026-07-10T09:00:00Z")
    outcomes = [
        process_pr(pr1, SOME_DIFF, [], frozenset(), []),
        process_pr(pr2, SOME_DIFF, [], frozenset(), []),
    ]
    data = build_queue_dashboard([pr1, pr2], outcomes)
    assert [item["pr"] for item in data["queue"]] == [1, 2]

    pr1_updated = _pr(number=1, updated_at="2026-07-10T10:00:00Z")
    data = build_queue_dashboard([pr1_updated, pr2], outcomes)
    assert [item["pr"] for item in data["queue"]] == [2, 1]


def test_queue_dashboard_write_skips_timestamp_only_churn():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY)
    outcome = process_pr(pr1, SOME_DIFF, [], frozenset(), [])
    data = build_queue_dashboard([pr1], [outcome])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.json")
        assert write_queue_dashboard(path, data)
        first = open(path, encoding="utf-8").read()
        assert not write_queue_dashboard(path, data)
        assert open(path, encoding="utf-8").read() == first


def test_run_once_continues_when_one_pr_write_back_fails():
    """A failed write-back on one PR (e.g. a transient GitHub 504 while
    labeling it) must not abort the sweep, fail an unrelated PR, or skip the
    dashboard publish -- the other PRs are still processed and the queue is
    still written."""
    import subprocess

    pr1 = _pr(number=1, author="alice", head_sha="s1")
    pr2 = _pr(number=2, author="bob", head_sha="s2")

    class BoomOnPR1(FakeClient):
        def add_label(self, pr_number, label):
            if pr_number == 1:
                raise subprocess.CalledProcessError(1, "gh", "", "504 gateway timeout")
            super().add_label(pr_number, label)

    client = BoomOnPR1(prs={"all": [pr1, pr2], "open": [pr1, pr2]},
                       diffs={1: SOME_DIFF, 2: SOME_DIFF})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.json")
        outcomes = run_once(client, dry_run=False, dashboard_data=path)
        # both PRs decided despite #1's write-back exploding
        assert {o.pr for o in outcomes} == {1, 2}
        # PR #2's write-back still happened, and the dashboard still published
        assert any(a[1] == 2 for a in client.actions)
        assert os.path.exists(path)


def test_is_transient_classifies_5xx_and_timeouts():
    from eval.github_client import _is_transient
    assert _is_transient("HTTP 504 Gateway Timeout")
    assert _is_transient("secondary rate limit exceeded")
    assert _is_transient("connection reset by peer")
    assert not _is_transient("could not find label 'nope'")
    assert not _is_transient("")


def test_exec_retries_transient_then_succeeds():
    import subprocess
    from unittest import mock

    from eval import github_client as gc

    calls = {"n": 0}

    def fake_run(cmd, capture_output, text):
        calls["n"] += 1
        if calls["n"] < 3:
            return subprocess.CompletedProcess(cmd, 1, "", "504 Gateway Timeout")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    with mock.patch.object(gc.subprocess, "run", fake_run), \
            mock.patch.object(gc.time, "sleep", lambda _s: None):
        gc.GitHubClient("z/r").add_label(1, "x")   # succeeds after 2 retries, no raise
    assert calls["n"] == 3


def test_exec_does_not_retry_non_transient():
    import subprocess
    from unittest import mock

    from eval import github_client as gc

    calls = {"n": 0}

    def fake_run(cmd, capture_output, text):
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 1, "", "label not found")

    raised = False
    with mock.patch.object(gc.subprocess, "run", fake_run), \
            mock.patch.object(gc.time, "sleep", lambda _s: None):
        try:
            gc.GitHubClient("z/r").add_label(1, "x")
        except subprocess.CalledProcessError:
            raised = True
    assert raised and calls["n"] == 1   # raised immediately, no wasted retries


def test_declared_track_parses_checked_box():
    from eval.pr_bot import declared_track
    body = "## Target track\n- [ ] full-rank\n- [x] low-rank\n- [ ] decaying-spectrum\n"
    assert declared_track(body) == "low-rank"
    assert declared_track("- [X] decaying-spectrum — smooth data") == "decaying-spectrum"
    assert declared_track("- [ ] full-rank\n- [ ] low-rank") is None   # none checked
    assert declared_track("") is None


def test_queue_record_carries_declared_track():
    from eval.pr_bot import _queue_record, GateOutcome
    pr = _pr(number=7, body="- [x] low-rank\n" + SCORECARD_BODY)
    rec = _queue_record(pr, GateOutcome(7, "eval_pending", kind="feat"))
    assert rec["track"] == "low-rank"
    # unspecified track -> None (gpu bot falls back to the full-rank reference)
    pr2 = _pr(number=8, body=SCORECARD_BODY)
    assert _queue_record(pr2, GateOutcome(8, "eval_pending", kind="feat"))["track"] is None


def test_declared_transform_parses_and_ignores_placeholder():
    from eval.pr_bot import declared_transform
    assert declared_transform("**Transform:** `nystrom`") == "nystrom"
    assert declared_transform("**transform:** dct") == "dct"
    assert declared_transform("**Transform:** `____`") is None    # unfilled placeholder
    assert declared_transform("no transform field here") is None


def test_queue_record_carries_declared_transform():
    from eval.pr_bot import _queue_record, GateOutcome
    pr = _pr(number=9, body="**Transform:** `nystrom`\n" + SCORECARD_BODY)
    rec = _queue_record(pr, GateOutcome(9, "eval_pending", kind="feat"))
    assert rec["transform"] == "nystrom"


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
