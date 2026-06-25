# Registering CCO on Gittensor (SN74) — operator playbook

This is the maintainer-side checklist to register `zeokin/Cuda-Compute-OSS` on Gittensor and
make it function as the CCO king-of-the-hill competition. It reflects the **actual SN74 validator
behaviour** (source: `github.com/entrius/gittensor`), not just the public "For Maintainers" page.

> Status of basic registration: **eligible now** — public ✓, MIT ✓, maintainer ✓, actively
> maintained ✓. What still has to be done before it works as the competition is below.

---

## 0. Do these by hand first (not automatable / outward-facing)

1. **Revoke the leaked GitHub token** embedded in the local clone's `git remote` URL, then reset the
   remote without a token (`git remote set-url origin https://github.com/zeokin/Cuda-Compute-OSS.git`).
   Per `SECURITY.md`, a leaked token must be **revoked**, not just rotated.
2. **Install the Gittensor read-only GitHub App** on the repo and **submit the gittensor.io
   registration form** (§4 has suggested answers).
3. In the manual review, raise the **custom-scoring asks** (§5) — the registry is team-curated; you
   cannot self-set `emission_share` or the label config.

---

## 1. What was already changed in-repo (done, verified against the entrius/gittensor source)

`cco.config.json` → `gittensor_repo_config` now uses the **real, source-verified** `RepositoryConfig`
field names (portable into Gittensor's `master_repositories.json`); `manifest.json` was regenerated +
re-verified (Gate-2 passes):

- `weight: 0.5` → **`emission_share: 0.01`** — placeholder; the **team sets** this. The validator
  enforces that the **sum across all repos is ≤ 1.0** (`load_weights.py`), and live repos run
  ~0.004–0.05, so 0.5 was never realistic.
- `eligibility_mode: false` → nested **`eligibility`** with zeros (`min_valid_merged_prs`,
  `min_credibility`, `min_valid_solved_issues`, `min_issue_credibility`) so a single crowned winner
  scores — defaults are `min_valid_merged_prs 3` / `min_credibility 0.80`, which would zero it.
- Added **`scoring.time_decay` flattened (`min_multiplier 1.0`)** + `pr_lookback_days 90`. Gittensor
  decays a merged PR's reward by **time-since-merge** by default (50% at 10d, 5% floor, out of the 30d
  lookback) — that erodes "earn while you hold the crown," so we flatten it. **Needs team approval** (§5).
- Unchanged (already correct): `trusted_label_pipeline: true`, the five `cco-winner-*`
  `label_multipliers`, `default_label_multiplier: 0.0`, `fixed_base_score: 1.0`.

> The `cco-winner-<track>` label is read from the **winning PR**, not the issue
> (`label_resolution.resolve_trusted_label_multiplier(pr.labels, …)`). The `cco-track` label goes on
> the **tracking issues**. The five `cco-winner-*` mirror the **current** (forward) tracks — add
> matching labels/issues/rows when BackwardBench lands.

### Clean entry to hand the Gittensor team for `master_repositories.json`

Match the file's existing structure (keyed by repo full name); strip the `_comment`; the team sets the
final `emission_share`:

```json
"zeokin/Cuda-Compute-OSS": {
  "emission_share": 0.01,
  "issue_discovery_share": 0.0,
  "maintainer_cut": 0.0,
  "trusted_label_pipeline": true,
  "default_label_multiplier": 0.0,
  "fixed_base_score": 1.0,
  "label_multipliers": {
    "cco-winner-rms_norm": 1.0,
    "cco-winner-matmul": 1.0,
    "cco-winner-qkv_part_rope": 1.0,
    "cco-winner-swiglu_input_quant": 1.0,
    "cco-winner-dsa_forward": 1.0
  },
  "eligibility": { "min_valid_merged_prs": 0, "min_credibility": 0.0, "min_valid_solved_issues": 0, "min_issue_credibility": 0.0 },
  "scoring": { "pr_lookback_days": 90, "time_decay": { "grace_period_hours": 8760, "sigmoid_midpoint_days": 36500, "sigmoid_steepness": 0.4, "min_multiplier": 1.0 } }
}
```

---

## 2. Create the labels (run these — they are public mutations, review first)

```bash
REPO=zeokin/Cuda-Compute-OSS

# King-of-the-hill winner labels (the CCO bot applies/strips these on the current champion PR).
for t in rms_norm matmul qkv_part_rope swiglu_input_quant dsa_forward; do
  gh label create "cco-winner-$t" --repo "$REPO" --color FFD700 \
    --description "Current $t champion (king-of-the-hill; earns emissions while held)" || \
  gh label edit   "cco-winner-$t" --repo "$REPO" --color FFD700 \
    --description "Current $t champion (king-of-the-hill; earns emissions while held)"
done

# Tag for the perpetual per-track competition issues.
gh label create "cco-track" --repo "$REPO" --color 1D76DB \
  --description "Standing per-track competition (perpetual issue; never closes on merge)"
```

---

## 3. Create the per-track tracking issues (gives Gittensor's "open issues" something to target)

Gittensor's model is "miners open PRs against your **open issues**." CCO has none, so create one
**perpetual** tracking issue per track. These intentionally **never close** (the contest reopens
with each new challenger) — flag that to the team (§5, Q5).

```bash
REPO=zeokin/Cuda-Compute-OSS

create_track_issue () {
  local track="$1" desc="$2"
  gh issue create --repo "$REPO" --label cco-track \
    --title "[track] $track — beat the current champion" \
    --body "$(cat <<EOF
**Standing king-of-the-hill competition for the \`$track\` kernel.** $desc

### How to compete
1. Start from the current champion: \`cp champions/$track/kernel.py kernel.py\`.
2. Optimize **only** \`kernel.py\` (Triton; no delegation — see CONTRIBUTING.md §3).
3. Self-score: \`uv run benchmark.py\` then \`uv run benchmark.py --blob\`.
4. Open a PR changing **only** \`kernel.py\` with the signed JSON payload (payload-schema.json),
   and reference this issue.

### How you win
- **Correctness is a hard gate** (5 stages vs the locked PyTorch oracle).
- The scored axis is **speedup vs the current champion** (Mann-Whitney U + margin), no more VRAM.
- Win → your PR becomes the champion, is merged, and gets the \`cco-winner-$track\` label; you
  earn emissions **while you hold the crown**. A new winner strips your label.

This issue is **perpetual** — it stays open as the standing ladder for \`$track\`. See
CONTRIBUTING.md and DESIGN.md for full rules and the threat model.
EOF
)"
}

create_track_issue rms_norm            "RMS normalization (memory-bound)."
create_track_issue matmul              "General matrix multiply / GEMM (compute-bound, tensor cores)."
create_track_issue qkv_part_rope       "Partial rotary position embedding (memory-bound)."
create_track_issue swiglu_input_quant  "SwiGLU activation + FP8 blockwise quant, multi-output (memory-bound)."
create_track_issue dsa_forward         "Causal GQA / FlashAttention forward, multi-output (compute-bound)."
```

---

## 4. Suggested gittensor.io registration-form answers

- **Repository:** `https://github.com/zeokin/Cuda-Compute-OSS`
- **Short description:** An objective, cheat-resistant GPU-kernel optimization competition. Miners
  submit one optimized Triton kernel per track; a locked harness verifies correctness against a
  PyTorch oracle and times it on a pinned GPU. A faster-and-still-correct kernel takes the crown
  (king-of-the-hill) and earns while it holds it. No subjective review.
- **Why it's a good fit for Gittensor:** scoring is fully automatable, objective, and
  cheat-resistant; merged PRs are exactly the unit Gittensor rewards; winning kernels are
  MIT-licensed and ship into production (incl. inference runtimes). Maps onto Gittensor's per-repo
  `RepositoryConfig` king-of-the-hill pattern (same shape as `anderdc/social-media-manager`).
- **Maintainer / contact:** zeokin (GitHub owner/admin).
- **Special requirements to flag:** (a) a custom `RepositoryConfig` (see §5); (b) the winning-PR
  **merge + `cco-winner-*` label is applied by our own maintainer bot**, never by the read-only App;
  (c) our per-track issues are **perpetual** (do not close on merge).

---

## 5. Questions / asks for the Gittensor team (manual review)

1. **Custom config approval.** Can our repo's `master_repositories.json` entry get
   `trusted_label_pipeline: true` + `label_multipliers{cco-winner-*: 1.0}` +
   `default_label_multiplier: 0.0` + `fixed_base_score: 1.0` + `eligibility{min_valid_merged_prs:0,
   min_credibility:0}`? (The validator source supports it; `anderdc/social-media-manager` runs an
   identical king-of-the-hill pattern.)
2. **emission_share + process.** What `emission_share` would you assign (we can't self-set the 0.5
   placeholder), and is the config a self-serve PR against `master_repositories.json` or team-edited?
3. **Field names.** Confirm we use `emission_share` (not `weight`) and a nested `eligibility` object
   (not an `eligibility_mode` boolean) — and that unknown top-level keys are ignored, not rejected.
4. **Bot-applied labels.** Will a `cco-winner-<track>` label applied by **our automated maintainer
   bot** (which may surface as `actor_association=NULL`) be honored under `trusted_label_pipeline:
   true`? If not granted, must the label be applied by an account in `MAINTAINER_ASSOCIATIONS`?
5. **Automation + perpetual issues.** Is a **fully-automated maintainer merge-bot** acceptable (it
   merges only the PR that passes our canonical GPU rerun + Mann-Whitney test and closes the rest),
   or do you require a human in the loop? And are **perpetual track issues** + `default_label_multiplier:0`
   (zero to every non-winner) acceptable for our repo?
6. **App vs PAT.** Does installing the read-only App **feed the validator mirror/scoring**, or must
   miners still register a fine-grained PAT with validators (`gitt miner post`) for their PRs to score?
7. **Identity attribution.** How is reward attributed — to the PR author's linked miner? We bind
   reward to a signed SN74 hotkey **inside the PR body**; we need both systems to credit the same
   party and to know which is authoritative for payout.
8. **Time-decay vs. holding the crown.** Your scorer decays a merged PR by time-since-merge (default
   ~50% at 10d, 5% floor, out of lookback at 30d). Our model is "earn **while** you hold the crown,"
   so we need the flattened `scoring.time_decay` (`min_multiplier: 1.0`) + `pr_lookback_days: 90`
   approved — otherwise a long-held champion's reward erodes. Is that override acceptable, or is the
   intended pattern "win = a decaying burst, re-win to top up"?

---

## 6. Still-to-build before the loop runs (not registration, but required to function)

- The **CCO maintainer bot** (own token, separate from the read-only App) that runs the gate
  pipeline, posts the verdict as a commit status, **merges only the winner + applies/strips the
  `cco-winner-*` label, and closes all other PRs** ("merged == crowned").
- The **canonical GPU rerun host** (Polaris) wired into that bot.
- The **on-chain SN74** identity + signature checks (Gate 1).

See the broader requirements doc / `DESIGN.md`. The merge/label attribution in `DESIGN.md §2-3` and
`cco/gate.py` should be reworded to credit the **CCO maintainer bot**, not any Gittensor-side merge.

---

## 7. How to add the CCO maintainer bot (it applies labels + merges)

The `cco-winner-<track>` label and the merge are performed by **your own** automation — never the
read-only Gittensor App. Source check (`label_resolution.py:53-70`, `constants.py:161`): a scoring
label counts if its actor is in `MAINTAINER_ASSOCIATIONS = ['OWNER', 'MEMBER', 'COLLABORATOR']`, **or**
for any actor (incl. a GitHub App that surfaces as `actor_association=NULL`) when
`trusted_label_pipeline: true` — which you have. Two equivalent ways to add it:

**Option A — bot user as a Collaborator (simplest, most robust).**
1. Create a dedicated GitHub machine account, e.g. `cco-judge-bot`.
2. Add it to `zeokin/Cuda-Compute-OSS` as a **Collaborator (Write)** (Settings → Collaborators &
   teams). As a `COLLABORATOR`, its labels count under `MAINTAINER_ASSOCIATIONS` even if
   `trusted_label_pipeline` were ever turned off.
3. Generate a **fine-grained PAT** owned by that account, scoped to **only this repo**, with
   *Contents: R/W, Pull requests: R/W, Issues: R/W*. Store it as an orchestrator/CI secret
   (e.g. `CCO_BOT_TOKEN`). Never commit it (cf. the leaked-token item in §0).

**Option B — a GitHub App you own.** Install your own App with `contents:write` + `pull_requests:write`
+ `issues:write`. Its labels surface as `actor_association=NULL`, so they rely on
`trusted_label_pipeline: true`.

**What the bot does** (your off-repo gate pipeline / a GitHub Action authenticated as the bot):
- on a **win** → `gh pr merge <n>`, `gh pr edit <n> --add-label cco-winner-<track>`, **strip the label
  from the previous champion PR**, and close the losing PRs;
- on a **loss** → close the PR and record the credibility decrement.

**Crown-holding nuance (verified in source):** scoring re-reads the **current** PR labels every ~2h
within `pr_lookback_days`, so king-of-the-hill works by **moving the single `cco-winner-<track>` label**
to the new champion PR. Combined with the flattened `time_decay` (§1/§5) and `pr_lookback_days: 90`,
a long-held champion keeps earning instead of decaying.

This bot is entirely separate from the read-only Gittensor App, which only **observes** the merged +
labeled result.
