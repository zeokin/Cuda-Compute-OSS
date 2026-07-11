"""Tests for eval.copycat_guard -- pure diff-text analysis, no GPU, no gh.

    python eval/tests/test_copycat_guard.py    (or)  python -m pytest eval/tests -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.copycat_guard import (
    check,
    check_one_pr,
    containment,
    fingerprint,
    is_docs_only,
    worst_verdict,
)
from eval.github_client import PRInfo

ORIGINAL_DIFF = """\
diff --git a/strategy/transforms.py b/strategy/transforms.py
--- a/strategy/transforms.py
+++ b/strategy/transforms.py
@@ -10,0 +11,10 @@
+class DCTTransform(Transform):
+    name = "dct"
+    def basis(self, n, m, backend, dtype, A=None, B=None):
+        xp = backend.xp
+        i = xp.arange(n, dtype=dtype).reshape(n, 1)
+        j = xp.arange(m, dtype=dtype).reshape(1, m)
+        Q = xp.cos((3.14159265 / n) * (i + 0.5) * j)
+        Q = Q * (2.0 / n) ** 0.5
+        return Q
+register_transform("dct", DCTTransform)
"""

# A near-verbatim copy (only the registration name renamed) -- should BLOCK
# (9/10 lines identical -> 0.90 containment, well above CONTAINMENT_BLOCK).
VERBATIM_COPY_DIFF = """\
diff --git a/strategy/transforms.py b/strategy/transforms.py
--- a/strategy/transforms.py
+++ b/strategy/transforms.py
@@ -10,0 +11,10 @@
+class DCTTransform(Transform):
+    name = "dct"
+    def basis(self, n, m, backend, dtype, A=None, B=None):
+        xp = backend.xp
+        i = xp.arange(n, dtype=dtype).reshape(n, 1)
+        j = xp.arange(m, dtype=dtype).reshape(1, m)
+        Q = xp.cos((3.14159265 / n) * (i + 0.5) * j)
+        Q = Q * (2.0 / n) ** 0.5
+        return Q
+register_transform("mydct", DCTTransform)
"""

# A partial rename (class/registration renamed + one identifier swapped on
# two lines) -- exact containment lands at 0.40 (below CONTAINMENT_WARN) but
# Levenshtein/bigram similarity stays high -- should WARN via the structural
# fallback, not be cleared just because literal containment is low.
STRUCTURAL_MATCH_DIFF = """\
diff --git a/strategy/transforms.py b/strategy/transforms.py
--- a/strategy/transforms.py
+++ b/strategy/transforms.py
@@ -10,0 +11,10 @@
+class MyDCT(Transform):
+    name = "mydct2"
+    def basis(self, n, m, backend, dtype, A=None, B=None):
+        xnp = backend.xp
+        i = xnp.arange(n, dtype=dtype).reshape(n, 1)
+        j = xnp.arange(m, dtype=dtype).reshape(1, m)
+        Q = xp.cos((3.14159265 / n) * (i + 0.5) * j)
+        Q = Q * (2.0 / n) ** 0.5
+        return Q
+register_transform("mydct2", MyDCT)
"""

# A genuinely independent transform touching the same file -- should be CLEAR.
INDEPENDENT_DIFF = """\
diff --git a/strategy/transforms.py b/strategy/transforms.py
--- a/strategy/transforms.py
+++ b/strategy/transforms.py
@@ -10,0 +11,12 @@
+class NystromTransform(Transform):
+    name = "nystrom"
+    def basis(self, n, m, backend, dtype, A=None, B=None):
+        xp = backend.xp
+        rng = xp.random.default_rng(self.seed)
+        idx = rng.choice(n, size=m, replace=False)
+        W = A[idx][:, idx]
+        C = A[:, idx]
+        Winv = xp.linalg.pinv(W)
+        Q, _ = xp.linalg.qr(C @ Winv)
+        return Q
+register_transform("nystrom", NystromTransform)
"""

# A different file entirely -- must never be flagged (no shared files).
UNRELATED_DIFF = """\
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,0 +2,2 @@
+Some unrelated documentation change.
+Nothing to do with transforms at all.
"""

DOCS_SUBTREE_DIFF = """\
diff --git a/docs/testing-strategy.md b/docs/testing-strategy.md
--- a/docs/testing-strategy.md
+++ b/docs/testing-strategy.md
@@ -1,0 +2,2 @@
+Explain the maintainer GPU window.
+Explain the public queue semantics.
"""


def test_verbatim_copy_is_blocked():
    orig = fingerprint(ORIGINAL_DIFF)
    copy = fingerprint(VERBATIM_COPY_DIFF)
    v = check(copy, orig)
    assert v.flagged and v.tier == "block", v


def test_structural_fallback_catches_low_containment_rework():
    orig = fingerprint(ORIGINAL_DIFF)
    reworked = fingerprint(STRUCTURAL_MATCH_DIFF)
    v = check(reworked, orig)
    assert v.flagged and v.tier == "warn" and "structural match" in v.reason, v


def test_independent_transform_is_clear():
    orig = fingerprint(ORIGINAL_DIFF)
    indep = fingerprint(INDEPENDENT_DIFF)
    v = check(indep, orig)
    assert not v.flagged and v.tier == "clear", v


def test_unrelated_file_is_clear_without_computing_similarity():
    orig = fingerprint(ORIGINAL_DIFF)
    unrelated = fingerprint(UNRELATED_DIFF)
    v = check(unrelated, orig)
    assert not v.flagged and v.reason == "no shared changed files"


def test_identical_diff_has_full_containment():
    orig = fingerprint(ORIGINAL_DIFF)
    same = fingerprint(ORIGINAL_DIFF)
    assert containment(same, orig) == 1.0


def test_is_docs_only_recognizes_root_and_docs_subtree_files():
    assert is_docs_only(fingerprint(UNRELATED_DIFF).files)
    assert is_docs_only(fingerprint(DOCS_SUBTREE_DIFF).files)
    assert not is_docs_only(fingerprint(ORIGINAL_DIFF).files)


def test_worst_verdict_picks_the_strongest_match_and_stops_at_block():
    orig = fingerprint(ORIGINAL_DIFF)
    copy = fingerprint(VERBATIM_COPY_DIFF)
    indep = fingerprint(INDEPENDENT_DIFF)
    # oldest-first order: independent PR first (clear), then the real match (block)
    author, v = worst_verdict(copy, [("author-a", indep), ("author-b", orig)])
    assert author == "author-b"
    assert v.tier == "block"


def test_worst_verdict_with_no_matches_is_clear():
    orig = fingerprint(ORIGINAL_DIFF)
    author, v = worst_verdict(orig, [])
    assert author is None and v.tier == "clear"


def test_empty_diff_does_not_crash():
    empty = fingerprint("")
    v = check(empty, fingerprint(ORIGINAL_DIFF))
    assert not v.flagged


# An added line whose source begins with "++" (e.g. C-style pre-increment) is
# emitted by git as "+++idx;". The parser must record it as content "++idx;",
# NOT mistake it for a file header and drop it.
PLUSPLUS_CONTENT_DIFF = """\
diff --git a/kernel.cu b/kernel.cu
--- a/kernel.cu
+++ b/kernel.cu
@@ -10,0 +11,3 @@
+    for (int idx = 0; idx < n; ) {
+++idx;
+    }
"""


def test_added_line_starting_with_plusplus_is_kept():
    fp = fingerprint(PLUSPLUS_CONTENT_DIFF)
    assert "kernel.cu" in fp.files
    assert "++idx;" in fp.added, f"++-prefixed added line was dropped: {fp.added}"


def test_plusplus_added_lines_count_toward_containment():
    # A copy that only shares the "++"-prefixed lines must not evade the guard by
    # having those lines silently discarded from the fingerprint.
    orig = fingerprint(PLUSPLUS_CONTENT_DIFF)
    copy = fingerprint(PLUSPLUS_CONTENT_DIFF)
    assert containment(copy, orig) == 1.0


class _FakeClientForOnePR:
    """Minimal fake covering just what check_one_pr() calls -- no subprocess."""

    def __init__(self, prs_all, diffs):
        self._prs_all = prs_all   # dict: pr_number -> PRInfo
        self._diffs = diffs       # dict: pr_number -> diff text

    def get_pr(self, pr_number):
        return self._prs_all[pr_number]

    def get_diff(self, pr_number):
        return self._diffs[pr_number]

    def list_prs(self, state="all"):
        return list(self._prs_all.values())


def test_check_one_pr_finds_an_earlier_copy():
    pr1 = PRInfo(number=1, title="t1", author="alice", is_draft=False, head_sha="s1")
    pr2 = PRInfo(number=2, title="t2", author="mallory", is_draft=False, head_sha="s2")
    client = _FakeClientForOnePR(
        prs_all={1: pr1, 2: pr2},
        diffs={1: ORIGINAL_DIFF, 2: VERBATIM_COPY_DIFF},
    )
    author, verdict = check_one_pr(client, 2)
    assert author == "alice" and verdict.tier == "block"


def test_check_one_pr_ignores_same_authors_earlier_pr():
    pr1 = PRInfo(number=1, title="t1", author="alice", is_draft=False, head_sha="s1")
    pr2 = PRInfo(number=2, title="t2", author="alice", is_draft=False, head_sha="s2")
    client = _FakeClientForOnePR(
        prs_all={1: pr1, 2: pr2},
        diffs={1: ORIGINAL_DIFF, 2: VERBATIM_COPY_DIFF},
    )
    author, verdict = check_one_pr(client, 2)
    assert verdict.tier == "clear"


def test_check_one_pr_skips_docs_only_prs():
    pr1 = PRInfo(number=1, title="t1", author="alice", is_draft=False, head_sha="s1")
    pr2 = PRInfo(number=2, title="t2", author="mallory", is_draft=False, head_sha="s2")
    client = _FakeClientForOnePR(
        prs_all={1: pr1, 2: pr2},
        diffs={1: ORIGINAL_DIFF, 2: UNRELATED_DIFF},
    )
    author, verdict = check_one_pr(client, 2)
    assert author is None
    assert verdict.tier == "clear"
    assert verdict.reason == "docs-only change"


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
