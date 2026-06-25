"""CCO gate-pipeline orchestrator (OFF-REPO).

This package is the automation that turns the locked CCO substrate into a live king-of-the-hill
competition on Bittensor SN74 (gittensor). It ingests a PR, walks the gates (default verdict =
reject), runs the canonical GPU rerun on Polaris, decides challenger-vs-champion via the in-repo
`cco.significance` test, and — as the **CCO maintainer bot** (a maintainer-owned token, NOT the
read-only Gittensor App) — merges the winner, moves the `cco-winner-<track>` label, and closes the
rest. SN74 validators then only observe the merged + labeled state.

It is meant to run OFF-REPO (a separate service / GitHub Action) against the byte-locked substrate;
it lives here as a skeleton so the loop is buildable and testable. External systems (Polaris GPU host,
GitHub writes, SN74 identity, rate limiting) are injected behind interfaces in `clients.py`, with
Mock* implementations so the whole pipeline runs end-to-end WITHOUT a GPU, a Polaris account, GitHub
write creds, or a chain connection. The decision logic uses the real `cco.*` modules.
"""

__all__ = ["models", "clients", "pipeline"]
