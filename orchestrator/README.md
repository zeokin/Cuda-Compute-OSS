# CCO orchestrator (off-repo gate pipeline) — skeleton

The automation that turns the locked CCO substrate into a live king-of-the-hill competition on
Bittensor SN74. It is designed to run **off-repo** (a separate service / GitHub Action) against the
byte-locked competition repo; it lives here as a runnable skeleton.

```
PR ─► parse ─► Gate1 identity ─► Gate2 manifest+diff ─► Gate3 static guard ─► rate-limit
                                                                                   │
                                              Gate4 canonical rerun on Polaris ◄────┘
                                       (challenger + champion, sealed, interleaved)
                                                                                   │
                          VRAM guard + cco.significance.challenger_wins ◄───────────┘
                                                                                   │
                 WIN ─► bot merges PR + moves cco-winner-<track> + closes losers   │
                 else ─► bot closes PR                                             ▼
                              SN74 validators only OBSERVE the merged+labeled state
```

## What's real vs. mocked

- **Real (in-repo `cco.*`):** the static no-delegation guard + size/track check (`cco.gate.gate3`),
  manifest integrity (`cco.manifest_tool.verify`), the PR-HEAD seed (`cco.seed.seed_from_sha`), and
  the challenger-vs-champion win test (`cco.significance.challenger_wins`).
- **Injected behind interfaces (`clients.py`), Mock* for now:**
  - `PolarisClient` — provisions a sealed pinned GPU and runs the locked scorer image → score blob.
  - `GitHubOps` — the **CCO maintainer bot** (a maintainer-owned token, **not** the read-only
    Gittensor App): merge / move `cco-winner-<track>` / close.
  - `IdentityVerifier` — Gate 1 (SN74 metagraph + sr25519/ed25519 signature + author==bound id).
  - `RateLimiter` — 1 rerun / hotkey / 24h.

## Run the demo (no GPU / creds needed)

```bash
python3 orchestrator/demo.py        # or: python3 -m orchestrator.demo
```

Drives four scenarios end-to-end: WIN (merge + crown move), LOSE (below the 5% margin),
REJECT-static (Gate 3 catches `torch.matmul`), REJECT-runtime (rerun reports vendor delegation).

## To productionize (replace the mocks)

1. **`PolarisClient`** → `POST /api/v2/compute/instances` on the pinned SKU → SSH →
   `docker run --gpus device=0 --network=none -v kernel.py:ro cco-runtime:<digest>` (clocks locked,
   exclusive GPU) → parse the `SCORE BLOB` from stdout → `DELETE` the instance. Handle 402/429/503 +
   create-timeout recovery-by-list.
2. **`GitHubOps`** → a bot account (owner/collaborator) or a GitHub App you own, with a fine-grained
   PAT (Contents/PRs/Issues R/W) scoped to this repo. Never the Gittensor App.
3. **`IdentityVerifier`** → bittensor metagraph membership + signature verification + the
   GitHub↔hotkey binding; enforce `pr.author == payload.github_user`.
4. **`RateLimiter`** → a durable per-hotkey store (a new PR resets the cooldown).
5. Run it on a schedule / PR webhook; freeze the PR (head SHA + body hash) between gate-walk and merge.

> This package is intentionally **not** part of the locked competition surface (`manifest.json` /
> `locked_paths`); it is maintainer infrastructure and may be split into its own repo for deployment.
