# Copycat detection & history

A **copycat** PR re-submits substantially the same diff as an earlier PR
(often an already-merged one) to farm credit for someone else's work. The
eval bot detects this automatically -- adapted from sparkinfer's proven
detector on SN74, see [entrius/gittensor's master_repositories.json](https://github.com/entrius/gittensor)
for how graduated enforcement is scored elsewhere on the subnet.

## How it works

- PRs are evaluated **oldest-first** (ascending PR number, see
  `eval.pr_bot.run_once`), so the original is always seen before any copy.
- For each open PR, the bot fingerprints the diff (`eval.copycat_guard.fingerprint`
  -- changed files + normalized non-comment added lines) and compares it
  against **every earlier PR** (open, closed, or merged) by a *different*
  author.
- Three graduated layers (`eval/copycat_guard.py`):
  1. **Containment >= 80%** -- instant block, zero tolerance. Labeled
     [`copycat`](../../labels/copycat), commented, closed.
  2. **Containment 70-79%** -- labeled
     [`copycat-warn`](../../labels/copycat-warn) and held for maintainer
     review, not auto-closed.
  3. **Structural match** (Levenshtein ratio >= 0.70 AND token-bigram cosine
     >= 0.60) even below 70% containment -- catches a lightly reworded copy.
     Also warned, not auto-closed.
- Self-resubmission is excluded: an author's own earlier PRs are never
  compared against their later ones.

The machine-readable log is [`copycats.json`](./copycats.json) -- one entry
per detected copycat `{pr, author, original, date, tier}`. Append-only,
maintained by the bot.

## Thresholds are provisional

These containment/similarity thresholds were carried over from sparkinfer,
which tunes them against multi-file CUDA kernel diffs. A CCO `Transform`
subclass is often only 10-30 lines, where independent-but-similar `rsvd`
variants can legitimately overlap heavily -- expect these numbers to need
retuning empirically once real submissions start arriving (see
the protected evaluator), not to hold as tuned truth from day one.

## History

*(empty -- no copycat has been detected yet)*

| date | copycat PR | author | copied from | tier | note |
|------|-----------|--------|-------------|------|------|
