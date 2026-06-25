"""The CCO gate-pipeline orchestrator state machine.

`Orchestrator.run_pr(pr)` walks the gates (default verdict = reject, short-circuit on first failure),
runs the canonical rerun via the injected Polaris client, decides the winner with the in-repo
`cco.significance` test, and performs the bot actions (merge / move label / close). It uses the REAL
`cco.*` enforcement modules; only Polaris / GitHub / identity / rate-limit are injected so it runs
without a GPU or write creds.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# Bootstrap: put the repo root on sys.path so `import cco.*` resolves whether this is imported as a
# package module or run standalone.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cco.gate import gate3                              # noqa: E402
from cco.seed import seed_from_sha                      # noqa: E402
from cco.significance import challenger_wins, load_thresholds_from_config  # noqa: E402
from cco import manifest_tool                           # noqa: E402

from .clients import GitHubOps, IdentityVerifier, PolarisClient, RateLimiter  # noqa: E402
from .models import Decision, GateResult, PullRequest    # noqa: E402


class _ShortCircuit(Exception):
    def __init__(self, verdict: str, reason: str):
        self.verdict = verdict
        self.reason = reason


@dataclass
class Orchestrator:
    repo_root: str
    config_path: str
    image_digest: str
    polaris: PolarisClient
    github: GitHubOps
    identity: IdentityVerifier
    rate_limiter: RateLimiter
    # track -> PR number currently holding cco-winner-<track> (in prod: read from labels / a store).
    champion_pr: dict = field(default_factory=dict)

    def __post_init__(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)
        self._tracks = set(self._config.get("tracks", []))
        self._vram_max_regression_pct = float(
            self._config.get("scoring", {}).get("guard_axes", {})
            .get("peak_vram_mb", {}).get("max_regression_pct", 5.0)
        )
        self._min_improvement_pct, self._p_threshold = load_thresholds_from_config(self.config_path)

    # ----------------------------------------------------------------- run
    def run_pr(self, pr: PullRequest) -> Decision:
        gates: list[GateResult] = []
        track = pr.payload.kernel_type
        significance: dict = {}
        actions: list[str] = []

        def gate(name: str, passed: bool, detail: str, fail_verdict: str = "reject") -> None:
            gates.append(GateResult(name, passed, detail))
            if not passed:
                raise _ShortCircuit(fail_verdict, f"{name}: {detail}")

        try:
            # 0. Parse / declared track
            gate("parse", track in self._tracks,
                 f"declared track {track!r}" if track in self._tracks else f"unknown track {track!r}")

            # 1. Identity (GitHub <-> hotkey <-> SN74 metagraph <-> signature)
            ok, detail = self.identity.verify(pr)
            gate("identity", ok, detail)

            # 2. Manifest integrity + diff-only-kernel.py
            changed = set(pr.changed_files)
            if changed != {"kernel.py"}:
                gate("diff", False, f"PR diff touches {sorted(changed - {'kernel.py'})} besides kernel.py")
            if pr.checkout_dir:
                with open(os.path.join(self.repo_root, "manifest.json"), "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                vios = manifest_tool.verify(pr.checkout_dir, manifest)
                gate("manifest", not vios,
                     "clean" if not vios else f"{len(vios)} violation(s): {vios[:3]}")
            else:
                gates.append(GateResult("manifest", True, "skipped (no checkout_dir in this run)"))

            # 3. Static no-delegation scan (+ size cap + declared-track match) — cheap, pre-GPU
            g3 = gate3(pr.kernel_path, track, self.config_path)
            gate("static_guard", g3["pass"], "clean" if g3["pass"] else "; ".join(g3["reasons"]))

            # 4. Rate limit (1 rerun / hotkey / 24h)
            gate("rate_limit", self.rate_limiter.allow(pr.payload.hotkey),
                 "ok" if True else "", fail_verdict="reject")

            # 5. Gate 4 — canonical rerun on Polaris (challenger + champion, interleaved, sealed)
            seed = seed_from_sha(pr.head_sha)
            champ_path = os.path.join(self.repo_root, "champions", track, "kernel.py")
            if not os.path.isfile(champ_path):
                raise _ShortCircuit("error", f"no champion kernel for track {track!r} at {champ_path}")
            with open(champ_path, "r", encoding="utf-8") as f:
                champ_src = f.read()
            with open(pr.kernel_path, "r", encoding="utf-8") as f:
                chal_src = f.read()

            champ = self.polaris.score(kernel_source=champ_src, kernel_type=track, seed=seed,
                                       image_digest=self.image_digest, role="champion")
            chal = self.polaris.score(kernel_source=chal_src, kernel_type=track, seed=seed,
                                      image_digest=self.image_digest, role="challenger")

            if chal.delegation:
                gate("rerun", False, f"delegation: {chal.delegation}", fail_verdict="reject")
            if not chal.correct:
                gate("rerun", False, "correctness FAIL (hard gate)", fail_verdict="lose")
            if chal.below_floor:
                gate("rerun", False, "below roofline floor (impossible/memoized timing)", fail_verdict="lose")
            gates.append(GateResult("rerun", True, f"correct; {len(chal.latencies_us)} blocks"))

            # 6. Decision — VRAM guard band + significance (challenger vs champion)
            vram_cap = champ.peak_vram_mb * (1.0 + self._vram_max_regression_pct / 100.0)
            vram_ok = chal.peak_vram_mb <= vram_cap
            gates.append(GateResult("vram", vram_ok,
                                    f"{chal.peak_vram_mb:.0f}/{vram_cap:.0f} MB cap"))

            significance = challenger_wins(champ.latencies_us, chal.latencies_us,
                                           min_improvement_pct=self._min_improvement_pct,
                                           p_threshold=self._p_threshold)
            won = bool(vram_ok and significance["win"])

            if won:
                # 7. Bot actions — merge, move the crown, record the new champion.
                prior = self.champion_pr.get(track)
                self.github.merge(pr.number)
                self.github.move_winner_label(track, to_pr=pr.number, from_pr=prior)
                self.champion_pr[track] = pr.number
                self.github.post_status(pr.number, True,
                                        f"WIN +{significance['improvement_pct']:.1f}% (lb +{significance['improvement_lb_pct']:.1f}%)")
                actions = list(getattr(self.github, "actions", []))
                return Decision(pr.number, track, "win", gates, significance, actions,
                                reason=f"beats champion by {significance['improvement_pct']:.1f}% (p={significance['p_value']:.4f})")
            else:
                reason = ("VRAM regression" if not vram_ok else
                          f"not a significant win (p={significance['p_value']:.4f}, "
                          f"lb +{significance['improvement_lb_pct']:.1f}% < {self._min_improvement_pct:.0f}%)")
                self.github.close(pr.number, reason)
                self.github.post_status(pr.number, False, reason)
                actions = list(getattr(self.github, "actions", []))
                return Decision(pr.number, track, "lose", gates, significance, actions, reason=reason)

        except _ShortCircuit as sc:
            if sc.verdict in ("reject", "lose"):
                self.github.close(pr.number, sc.reason)
                self.github.post_status(pr.number, False, sc.reason)
            actions = list(getattr(self.github, "actions", []))
            return Decision(pr.number, track, sc.verdict, gates, significance, actions, reason=sc.reason)
