"""Injected external clients for the orchestrator.

Each external system the orchestrator touches is behind an interface so the pipeline can run and be
tested with the Mock* implementations (no GPU / Polaris / GitHub creds / chain needed). Swap in the
real implementations for production — the pipeline code does not change.
"""

from __future__ import annotations

import abc
import hashlib
import random
import time
from typing import Optional

from .models import PullRequest, ScoreBlob


# ===========================================================================
# Polaris — the GPU scoring host (Gate 4 canonical rerun)
# ===========================================================================

class PolarisClient(abc.ABC):
    """Provision a sealed, pinned GPU; run the locked scorer image; return the bound score blob.

    Production impl (TODO): POST /api/v2/compute/instances on the pinned SKU -> SSH ->
    `docker run --gpus device=0 --network=none -v kernel.py:ro cco-runtime:<image_digest>` (clocks
    locked, exclusive GPU) -> parse the SCORE BLOB from stdout -> DELETE the instance. Handle
    402/429/503 + create-timeout recovery-by-list.
    """

    @abc.abstractmethod
    def score(self, *, kernel_source: str, kernel_type: str, seed: int,
              image_digest: str, role: str = "challenger") -> ScoreBlob:
        ...


class MockPolarisClient(PolarisClient):
    """Synthetic scorer for testing the loop. `champion_latency_us` + `challenger_speedup` force a
    win/lose; `challenger_correct` / `challenger_delegation` simulate a kernel that trips the runtime
    trap. Deterministic given (role, seed)."""

    def __init__(self, champion_latency_us: float = 100.0, challenger_speedup: float = 1.10,
                 cov: float = 0.02, n_blocks: int = 30, vram_mb: float = 512.0,
                 challenger_correct: bool = True, challenger_delegation: Optional[str] = None,
                 challenger_below_floor: bool = False):
        self.champion_latency_us = champion_latency_us
        self.challenger_speedup = challenger_speedup
        self.cov = cov
        self.n_blocks = n_blocks
        self.vram_mb = vram_mb
        self.challenger_correct = challenger_correct
        self.challenger_delegation = challenger_delegation
        self.challenger_below_floor = challenger_below_floor

    def score(self, *, kernel_source: str, kernel_type: str, seed: int,
              image_digest: str, role: str = "challenger") -> ScoreBlob:
        rng = random.Random(f"{seed}:{role}")
        is_champ = role == "champion"
        base = self.champion_latency_us if is_champ else self.champion_latency_us / self.challenger_speedup
        lat = [max(0.1, rng.gauss(base, base * self.cov)) for _ in range(self.n_blocks)]
        correct = True if is_champ else self.challenger_correct
        delegation = None if is_champ else self.challenger_delegation
        below = False if is_champ else self.challenger_below_floor
        digest = hashlib.sha256(f"{role}:{kernel_type}:{seed}:{kernel_source[:64]}".encode()).hexdigest()
        return ScoreBlob(kernel_type=kernel_type, correct=correct, latencies_us=lat,
                         peak_vram_mb=self.vram_mb, delegation=delegation, below_floor=below,
                         blob_sha256=digest)


# ===========================================================================
# GitHub write ops — performed by the CCO MAINTAINER BOT (never the Gittensor App)
# ===========================================================================

class GitHubOps(abc.ABC):
    """The bot's repo write surface. Production impl: `gh`/REST authenticated as a maintainer-owned
    bot account (owner/collaborator) or a GitHub App you own with contents+PR+issues write."""

    @abc.abstractmethod
    def merge(self, pr_number: int) -> None:
        ...

    @abc.abstractmethod
    def move_winner_label(self, track: str, to_pr: int, from_pr: Optional[int]) -> None:
        """Strip `cco-winner-<track>` from the prior champion PR (if any) and add it to `to_pr`."""
        ...

    @abc.abstractmethod
    def close(self, pr_number: int, reason: str) -> None:
        ...

    @abc.abstractmethod
    def post_status(self, pr_number: int, ok: bool, summary: str) -> None:
        ...


class MockGitHubOps(GitHubOps):
    """Records actions instead of performing them (for tests / dry runs)."""

    def __init__(self):
        self.actions: list[str] = []

    def merge(self, pr_number: int) -> None:
        self.actions.append(f"merge PR#{pr_number}")

    def move_winner_label(self, track: str, to_pr: int, from_pr: Optional[int]) -> None:
        if from_pr is not None:
            self.actions.append(f"remove-label cco-winner-{track} from PR#{from_pr}")
        self.actions.append(f"add-label cco-winner-{track} to PR#{to_pr}")

    def close(self, pr_number: int, reason: str) -> None:
        self.actions.append(f"close PR#{pr_number} ({reason})")

    def post_status(self, pr_number: int, ok: bool, summary: str) -> None:
        self.actions.append(f"status PR#{pr_number} {'success' if ok else 'failure'}: {summary}")


# ===========================================================================
# SN74 identity (Gate 1)
# ===========================================================================

class IdentityVerifier(abc.ABC):
    @abc.abstractmethod
    def verify(self, pr: PullRequest) -> tuple[bool, str]:
        """Return (ok, detail). Production: hotkey on the SN74 metagraph + sr25519/ed25519 signature
        over commit_sha:kernel_sha256:kernel_type + github_user is the bound identity AND == pr.author."""
        ...


class MockIdentityVerifier(IdentityVerifier):
    """Trusts the payload; only enforces author == payload.github_user (the one check that needs no chain)."""

    def __init__(self, accept: bool = True):
        self.accept = accept

    def verify(self, pr: PullRequest) -> tuple[bool, str]:
        if not self.accept:
            return False, "identity rejected (mock)"
        if pr.author != pr.payload.github_user:
            return False, f"PR author {pr.author!r} != payload github_user {pr.payload.github_user!r}"
        return True, "identity ok (mock: signature/metagraph not checked)"


# ===========================================================================
# Rate limit (1 rerun / hotkey / 24h)
# ===========================================================================

class RateLimiter(abc.ABC):
    @abc.abstractmethod
    def allow(self, hotkey: str) -> bool:
        ...


class InMemoryRateLimiter(RateLimiter):
    """Per-hotkey cooldown. Production: a durable store keyed by hotkey; a new PR resets the cooldown."""

    def __init__(self, window_hours: float = 24.0, _clock=time.time):
        self.window_s = window_hours * 3600.0
        self._last: dict[str, float] = {}
        self._clock = _clock

    def allow(self, hotkey: str) -> bool:
        now = self._clock()
        last = self._last.get(hotkey)
        if last is not None and (now - last) < self.window_s:
            return False
        self._last[hotkey] = now
        return True
