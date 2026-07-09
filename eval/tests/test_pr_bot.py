"""Tests for eval.pr_bot's gate chain -- pure decision logic, no GitHub I/O.

process_pr() takes already-fetched data and performs no I/O itself, so every
test here uses plain fixtures; run_once()/FakeClient below exercise the
orchestration layer with a fake in-memory GitHub, never real `gh` calls.

    python eval/tests/test_pr_bot.py        (or)   python -m pytest eval/tests -q
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval import copycat_guard
from eval.pr_bot import (
    GPU_QUEUE_LABEL,
    NEEDS_PR_KIND_LABEL,
    NEEDS_SCORECARD_MARKER,
    PROTECTED_PATH_LABEL,
    READY_NON_GPU_LABEL,
    PRInfo,
    already_evaluated,
    already_queued,
    already_notified,
    build_queue_dashboard,
    changed_files,
    has_scorecard,
    process_pr,
    run_once,
    protected_paths,
    write_queue_dashboard,
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


def _pr(number=1, author="alice", is_draft=False, head_sha="sha1", body=SCORECARD_BODY):
    return PRInfo(number=number, title=f"feat: PR {number}", author=author,
                 is_draft=is_draft, head_sha=head_sha, body=body)


def test_draft_is_skipped():
    out = process_pr(_pr(is_draft=True), SOME_DIFF, [], frozenset(), [])
    assert out.action == "skip_draft"


def test_blocked_contributor_is_closed():
    out = process_pr(_pr(author="badactor"), SOME_DIFF, [], frozenset({"badactor"}), [])
    assert out.action == "close_blocked"


def test_changed_files_from_diff():
    assert changed_files(SOME_DIFF) == frozenset({"strategy/transforms.py"})


def test_protected_paths_are_detected():
    assert protected_paths(PROTECTED_DIFF) == ("eval/evaluator.py",)


def test_protected_path_is_not_queued():
    out = process_pr(_pr(author="alice"), PROTECTED_DIFF, [], frozenset(), [])
    assert out.action == "protected_path"
    assert out.label == PROTECTED_PATH_LABEL


def test_unknown_pr_kind_is_flagged():
    pr = PRInfo(number=1, title="PR 1", author="alice", is_draft=False, head_sha="sha1",
                body="Just some changes.")
    out = process_pr(pr, "diff --git a/matmul/x.py b/matmul/x.py\n+++ b/matmul/x.py\n+pass\n",
                     [], frozenset(), [])
    assert out.action == "needs_pr_kind"
    assert out.label == NEEDS_PR_KIND_LABEL


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
    assert out.action == "needs_scorecard"
    assert out.label == "status:needs-scorecard"


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


def test_already_notified_helper():
    assert already_notified(["x", "<!-- cco-needs-scorecard:abc -->"], NEEDS_SCORECARD_MARKER, "abc")
    assert not already_notified(["x"], NEEDS_SCORECARD_MARKER, "abc")


class FakeClient:
    """In-memory stand-in for GitHubClient -- no subprocess/network calls."""

    def __init__(self, prs, diffs, comments=None):
        self._prs = prs               # dict: state -> list[PRInfo]
        self._diffs = diffs           # dict: pr_number -> diff text
        self._comments = comments or {}
        self.actions = []             # records of what WOULD have been written

    def list_prs(self, state="open"):
        if state == "all":
            return self._prs.get("all", [])
        return self._prs.get(state, [])

    def get_diff(self, pr_number):
        return self._diffs.get(pr_number, "")

    def get_comments(self, pr_number):
        return self._comments.get(pr_number, [])

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
    # "needs_scorecard" path instead by using a body with no scorecard, so we
    # can observe a real write action deterministically without touching the
    # filesystem-backed blocked list.
    pr1.body = NO_SCORECARD_BODY
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "needs_scorecard"
    assert ("add_label", 1, "status:needs-scorecard") in client.actions
    assert any(a[0] == "post_comment" for a in client.actions)


def test_run_once_live_mode_does_not_repeat_needs_scorecard_comment():
    pr1 = _pr(number=1, author="alice", body=NO_SCORECARD_BODY)
    marker = NEEDS_SCORECARD_MARKER.format(sha="sha1")
    client = FakeClient(
        prs={"all": [pr1], "open": [pr1]},
        diffs={1: SOME_DIFF},
        comments={1: [marker]},
    )
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "needs_scorecard"
    assert ("add_label", 1, "status:needs-scorecard") in client.actions
    assert not any(a[0] == "post_comment" for a in client.actions)


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


def test_run_once_live_mode_labels_protected_path():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY)
    client = FakeClient(prs={"all": [pr1], "open": [pr1]}, diffs={1: PROTECTED_DIFF})
    outcomes = run_once(client, dry_run=False)
    assert outcomes[0].action == "protected_path"
    assert ("add_label", 1, PROTECTED_PATH_LABEL) in client.actions


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


def test_queue_dashboard_orders_ready_prs_oldest_first():
    pr1 = _pr(number=1, author="alice", body=SCORECARD_BODY)
    pr2 = PRInfo(number=2, title="fix: PR 2", author="bob", is_draft=False, head_sha="sha1",
                 body=NO_SCORECARD_BODY)
    pr3 = _pr(number=3, author="carol", body=SCORECARD_BODY)
    outcomes = [
        process_pr(pr1, SOME_DIFF, [], frozenset(), []),
        process_pr(pr2, SOME_DIFF, [], frozenset(), []),
        process_pr(pr3, SOME_DIFF, [], frozenset(), []),
    ]
    data = build_queue_dashboard([pr1, pr2, pr3], outcomes)
    assert [item["pr"] for item in data["queue"]] == [1, 3]
    assert [item["position"] for item in data["queue"]] == [1, 2]
    assert len(data["open_prs"]) == 3


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
