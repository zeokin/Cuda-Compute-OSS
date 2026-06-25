"""Data models for the CCO gate-pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Payload:
    """The fenced JSON payload a miner puts in the PR body (see payload-schema.json).

    Gate 1 verifies that `signature` (over `commit_sha:kernel_sha256:kernel_type`) checks out under
    `hotkey`, that `hotkey` is registered on the SN74 metagraph, and that `github_user` is the bound
    identity for that hotkey AND equals the PR author.
    """

    hotkey: str
    github_user: str
    kernel_type: str
    kernel_sha256: str
    commit_sha: str
    signature: str

    REQUIRED = ("hotkey", "github_user", "kernel_type", "kernel_sha256", "commit_sha", "signature")

    @classmethod
    def from_dict(cls, d: dict) -> "Payload":
        missing = [k for k in cls.REQUIRED if not d.get(k)]
        if missing:
            raise ValueError(f"payload missing required field(s): {', '.join(missing)}")
        return cls(**{k: d[k] for k in cls.REQUIRED})


@dataclass
class PullRequest:
    """A submitted PR, as the orchestrator sees it (the bot reads this via read-only GitHub metadata)."""

    number: int
    repo: str
    author: str
    head_sha: str
    changed_files: list[str]
    payload: Payload
    # Absolute path to the challenger kernel.py at the PR HEAD (the bot checks the PR out).
    kernel_path: str
    # Absolute path to the PR-HEAD checkout root (for the Gate-2 manifest verify). Optional in tests.
    checkout_dir: Optional[str] = None


@dataclass
class ScoreBlob:
    """The bound score blob `benchmark.py --blob` emits for ONE kernel (challenger or champion).

    The orchestrator runs this twice per PR (challenger + champion, fresh & interleaved in one sealed
    Polaris job) and compares the two `latencies_us` samples.
    """

    kernel_type: str
    correct: bool
    latencies_us: list[float]
    peak_vram_mb: float
    delegation: Optional[str] = None  # non-None => disqualified (ran a banned vendor op)
    below_floor: bool = False         # True => physically-impossible (memoized) GPU time
    blob_sha256: str = ""


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Decision:
    """The orchestrator's verdict for one PR, plus the bot actions it took."""

    pr_number: int
    track: Optional[str]
    verdict: str  # "win" | "lose" | "reject" | "error"
    gates: list[GateResult] = field(default_factory=list)
    significance: dict = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    reason: str = ""

    def summary(self) -> str:
        gline = "  ".join(f"{g.name}={'ok' if g.passed else 'X'}" for g in self.gates)
        return f"PR#{self.pr_number} [{self.track}] -> {self.verdict.upper()}  ({gline})  {self.reason}".rstrip()
