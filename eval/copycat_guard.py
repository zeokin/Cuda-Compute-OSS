"""Copycat detection: does a PR's diff substantially reproduce an earlier
PR's diff? Adapted from sparkinfer's ``eval/copycat_guard.py`` -- the
comparison itself is diff-text analysis and carries over unchanged; only the
enforcement plumbing (which repo, how strike state is persisted) is
CCO-specific and lives in ``eval/pr_bot.py`` instead, which owns
``.github/blocked-contributors.txt``/``.github/copycats.json`` state.

Three graduated layers, checked in order:
  1. Exact containment >= CONTAINMENT_BLOCK -- instant block, zero tolerance.
  2. Exact containment >= CONTAINMENT_WARN  -- warn; strikes accumulate
     (strike-counting itself is pr_bot.py's job, not this module's).
  3. Structural similarity (Levenshtein ratio + bigram cosine) even below the
     containment thresholds -- catches lightly reworded copies.

Everything above :func:`main` is a pure function of diff text -- no GitHub
I/O, no GPU, fully unit-testable. :func:`main` is the thin, real-time,
single-PR entrypoint that .github/workflows/copycat-guard.yml invokes right
when a PR is opened (checked out at ``main``, never the PR's own code --
mirrors sparkinfer's copycat-guard.yml). It only labels/comments/closes;
the periodic full gate chain (draft/blocked/scorecard/copycat together) is
eval.pr_bot's job, run separately (Phase 3).

NOTE (flagged in docs/sn74-emission-strategy.md and the update plan): these
thresholds were tuned on sparkinfer's multi-file CUDA kernel diffs. A CCO
``Transform`` subclass is often 10-30 lines, where independent-but-similar
``rsvd`` variants can legitimately overlap heavily -- expect to retune
empirically once real submissions arrive, not to trust these numbers blindly.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from .github_client import GitHubClient

CONTAINMENT_BLOCK = 0.80
CONTAINMENT_WARN = 0.70
STRUCTURAL_FLOOR = 0.40      # below this containment, don't bother checking structural similarity
LEVENSHTEIN_THRESHOLD = 0.70
BIGRAM_COSINE_THRESHOLD = 0.60
DOC_ONLY_EXACT = frozenset({"README.md", "CONTRIBUTING.md", "BENCHMARKS.md", "LICENSE"})
DOC_ONLY_PREFIXES = ("docs/",)

_COMMENT_RE = re.compile(r"^\s*#")
_TIER_RANK = {"clear": 0, "warn": 1, "block": 2}


@dataclass
class Fingerprint:
    files: frozenset
    added: frozenset  # normalized non-comment added lines


@dataclass
class Verdict:
    flagged: bool
    tier: str  # "block" | "warn" | "clear"
    containment_ratio: float
    reason: str


def fingerprint(diff_text: str) -> Fingerprint:
    """Fingerprint a unified diff: changed files + normalized added lines.
    Blank lines and comment-only lines are excluded -- they inflate overlap
    without carrying any actual logic."""
    files: set = set()
    added: set = set()
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null":
                files.add(path)
            continue
        # Only the genuine "--- a/path" removal header (trailing space) is a header;
        # the "+++ b/path" add header is already consumed above. A bare "+++"/"---"
        # test here would also swallow ADDED CONTENT that starts with "++"/"--":
        # git prefixes an added line "++idx;" as "+++idx;", so dropping it here would
        # silently omit real added lines from the fingerprint and skew containment.
        if line.startswith("--- "):
            continue
        if line.startswith("+"):
            content = line[1:].strip()
            if content and not _COMMENT_RE.match(content):
                added.add(content)
    return Fingerprint(files=frozenset(files), added=frozenset(added))


def is_docs_only(files: frozenset) -> bool:
    return bool(files) and all(
        path in DOC_ONLY_EXACT or path.startswith(DOC_ONLY_PREFIXES)
        for path in files
    )


def containment(candidate: Fingerprint, original: Fingerprint) -> float:
    """Fraction of ``candidate``'s added lines that already appear in
    ``original``'s added lines. 0 if the candidate added nothing."""
    if not candidate.added:
        return 0.0
    return len(candidate.added & original.added) / len(candidate.added)


def _tokenize(added: frozenset) -> list:
    text = "\n".join(sorted(added))
    return re.findall(r"\w+", text)


def levenshtein_ratio(candidate: Fingerprint, original: Fingerprint) -> float:
    """difflib's SequenceMatcher ratio over the joined, sorted added-line
    text -- order-independent, robust to line reordering."""
    a = "\n".join(sorted(candidate.added))
    b = "\n".join(sorted(original.added))
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def bigram_cosine(candidate: Fingerprint, original: Fingerprint) -> float:
    """Cosine similarity of token-bigram frequency vectors -- catches a
    reworded copy that shares structure but few literal lines."""
    ta, tb = _tokenize(candidate.added), _tokenize(original.added)
    if len(ta) < 2 or len(tb) < 2:
        return 0.0
    bigrams_a = Counter(zip(ta, ta[1:]))
    bigrams_b = Counter(zip(tb, tb[1:]))
    common = set(bigrams_a) & set(bigrams_b)
    dot = sum(bigrams_a[k] * bigrams_b[k] for k in common)
    norm_a = sum(v * v for v in bigrams_a.values()) ** 0.5
    norm_b = sum(v * v for v in bigrams_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def shares_a_file(candidate: Fingerprint, original: Fingerprint) -> bool:
    return bool(candidate.files & original.files)


def check(candidate: Fingerprint, original: Fingerprint) -> Verdict:
    """Compare one candidate PR's fingerprint against ONE earlier PR's
    fingerprint. Callers should call this against every earlier PR (oldest
    first, excluding the candidate's own author -- self-resubmission is not
    copying) and keep the worst verdict via :func:`worst_verdict`."""
    if not shares_a_file(candidate, original):
        return Verdict(False, "clear", 0.0, "no shared changed files")

    c = containment(candidate, original)

    if c >= CONTAINMENT_BLOCK:
        return Verdict(True, "block", c,
                       f"{c:.0%} of added lines already appear in an earlier PR")
    if c >= CONTAINMENT_WARN:
        return Verdict(True, "warn", c,
                       f"{c:.0%} containment -- below the block threshold "
                       f"but a real overlap")
    if c >= STRUCTURAL_FLOOR:
        lev = levenshtein_ratio(candidate, original)
        bg = bigram_cosine(candidate, original)
        if lev >= LEVENSHTEIN_THRESHOLD and bg >= BIGRAM_COSINE_THRESHOLD:
            return Verdict(True, "warn", c,
                           f"structural match (levenshtein={lev:.2f}, "
                           f"bigram_cosine={bg:.2f}) despite only {c:.0%} "
                           f"literal containment")
    return Verdict(False, "clear", c, "no significant overlap")


def worst_verdict(candidate: Fingerprint, originals: list) -> tuple:
    """Check ``candidate`` against every ``(author, fingerprint)`` in
    ``originals`` and return the strongest match found:
    ``(matched_author_or_None, Verdict)``. Stops early once a "block" is
    found (nothing can outrank it)."""
    best_author = None
    best_verdict = Verdict(False, "clear", 0.0, "no earlier PR to compare against")
    for author, orig_fp in originals:
        v = check(candidate, orig_fp)
        if _TIER_RANK[v.tier] > _TIER_RANK[best_verdict.tier]:
            best_verdict, best_author = v, author
            if v.tier == "block":
                break
    return best_author, best_verdict


def check_one_pr(client: GitHubClient, pr_number: int) -> tuple:
    """Real-time check: fingerprint ``pr_number`` and compare it against
    every earlier PR (any state) by a different author. Returns
    ``(matched_author_or_None, Verdict)``. Read-only -- callers decide what
    to do with the result (see :func:`main`)."""
    candidate_pr = client.get_pr(pr_number)
    candidate_fp = fingerprint(client.get_diff(pr_number))
    if is_docs_only(candidate_fp.files):
        return None, Verdict(False, "clear", 0.0, "docs-only change")

    earlier = [p for p in client.list_prs("all") if p.number < pr_number]
    originals = [
        (p.author, fingerprint(client.get_diff(p.number)))
        for p in earlier if p.author != candidate_pr.author
    ]
    return worst_verdict(candidate_fp, originals)


def main(argv=None) -> int:
    """Entrypoint for .github/workflows/copycat-guard.yml -- one PR, checked
    right when it's opened. Deliberately narrow: label/comment on a "warn"
    verdict (leave the call to a maintainer), label/comment/close on "block"
    (zero tolerance, matches sparkinfer). Never touches the blocked-
    contributors list or strike counting -- that stateful bookkeeping is
    eval.pr_bot's job on its periodic sweep, not this fast real-time hook.
    """
    p = argparse.ArgumentParser(prog="python -m eval.copycat_guard")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    p.add_argument("--pr", type=int,
                   default=int(os.environ.get("PR_NUMBER", "0")) or None)
    args = p.parse_args(argv)
    if not args.repo or not args.pr:
        p.error("--repo and --pr are required (or GITHUB_REPOSITORY/PR_NUMBER env vars)")

    client = GitHubClient(args.repo)
    author, verdict = check_one_pr(client, args.pr)

    if verdict.tier == "block":
        client.add_label(args.pr, "copycat")
        client.post_comment(
            args.pr,
            f"Closed as a copycat submission: {verdict.reason} "
            f"(matches an earlier PR by {author}). See .github/COPYCATS.md.",
        )
        client.close_pr(args.pr, "copycat")
        print(f"PR #{args.pr}: block -- matches {author}: {verdict.reason}")
    elif verdict.tier == "warn":
        client.add_label(args.pr, "copycat-warn")
        client.post_comment(
            args.pr,
            f"Flagged for maintainer review: {verdict.reason} "
            f"(matches an earlier PR by {author}). See .github/COPYCATS.md.",
        )
        print(f"PR #{args.pr}: warn -- matches {author}: {verdict.reason}")
    else:
        print(f"PR #{args.pr}: clear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
