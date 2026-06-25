#!/usr/bin/env python3
"""End-to-end mock demo of the CCO orchestrator — runs WITHOUT a GPU/Polaris/GitHub creds.

It drives `Orchestrator.run_pr` through the real `cco.*` gates (static guard, manifest verify,
seed, significance) with Polaris + GitHub mocked, across four scenarios:
  1. WIN          — clean kernel, challenger 10% faster -> merged + crown moved
  2. LOSE         — clean kernel, only ~2% faster (below the 5% margin) -> closed
  3. REJECT(static)  — delegating kernel (torch.matmul) caught by Gate 3 before any GPU spend
  4. REJECT(runtime) — clean statically, but the rerun reports vendor delegation (popped trap / cuBLAS)

Run:  python3 orchestrator/demo.py   (or:  python3 -m orchestrator.demo)
"""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from orchestrator.clients import (InMemoryRateLimiter, MockGitHubOps, MockIdentityVerifier,  # noqa: E402
                                  MockPolarisClient)
from orchestrator.models import Payload, PullRequest  # noqa: E402
from orchestrator.pipeline import Orchestrator         # noqa: E402

CONFIG = os.path.join(_REPO_ROOT, "cco.config.json")
CLEAN_KERNEL = os.path.join(_REPO_ROOT, "champions", "rms_norm", "kernel.py")  # a real, clean kernel
IMAGE_DIGEST = "sha256:demo-unbuilt-image"

_DELEGATING_KERNEL = '''import torch
KERNEL_TYPE = "rms_norm"
def kernel_fn(x, weight, eps=1e-6):
    return torch.matmul(x, weight)   # vendor BLAS delegation -> Gate 3 must reject
'''


def _make_pr(number: int, kernel_path: str, head_sha: str, author: str = "miner-alice") -> PullRequest:
    payload = Payload(hotkey=f"5F...{number}", github_user=author, kernel_type="rms_norm",
                      kernel_sha256="deadbeef" * 8, commit_sha=head_sha, signature="0xSIG")
    return PullRequest(number=number, repo="zeokin/Cuda-Compute-OSS", author=author, head_sha=head_sha,
                       changed_files=["kernel.py"], payload=payload, kernel_path=kernel_path,
                       checkout_dir=_REPO_ROOT)


def _run(title: str, pr: PullRequest, polaris, champion_pr=None):
    gh = MockGitHubOps()
    orch = Orchestrator(repo_root=_REPO_ROOT, config_path=CONFIG, image_digest=IMAGE_DIGEST,
                        polaris=polaris, github=gh, identity=MockIdentityVerifier(),
                        rate_limiter=InMemoryRateLimiter(), champion_pr=dict(champion_pr or {}))
    decision = orch.run_pr(pr)
    print(f"\n### {title}")
    print(decision.summary())
    if decision.significance:
        s = decision.significance
        print(f"    significance: win={s['win']} p={s['p_value']:.4f} "
              f"point=+{s['improvement_pct']:.1f}% ci_lb=+{s['improvement_lb_pct']:.1f}%")
    for a in gh.actions:
        print(f"    bot> {a}")


def main() -> int:
    sha = "1a2b3c4d5e6f70819a0b1c2d3e4f5061728394a5"
    print("=" * 72)
    print("CCO orchestrator — end-to-end MOCK demo (no GPU / Polaris / GitHub creds)")
    print("=" * 72)

    # 1. WIN — clean kernel, 10% faster; a prior champion PR#7 holds the crown -> it gets moved.
    _run("1. WIN (clean, +10%)",
         _make_pr(101, CLEAN_KERNEL, sha[:39] + "1"),
         MockPolarisClient(challenger_speedup=1.10), champion_pr={"rms_norm": 7})

    # 2. LOSE — clean kernel, only ~2% faster (significant maybe, but below the 5% margin).
    _run("2. LOSE (clean, +2% < margin)",
         _make_pr(102, CLEAN_KERNEL, sha[:39] + "2"),
         MockPolarisClient(challenger_speedup=1.02, cov=0.01))

    # 3. REJECT (static) — delegating kernel caught by Gate 3 (never reaches the GPU).
    with tempfile.TemporaryDirectory() as td:
        kp = os.path.join(td, "kernel.py")
        with open(kp, "w", encoding="utf-8") as f:
            f.write(_DELEGATING_KERNEL)
        _run("3. REJECT static (torch.matmul)",
             _make_pr(103, kp, sha[:39] + "3"),
             MockPolarisClient(challenger_speedup=2.0))  # would be "fast" but never runs

    # 4. REJECT (runtime) — clean statically, but the rerun trips the vendor-symbol trap.
    _run("4. REJECT runtime (popped-trap cuBLAS)",
         _make_pr(104, CLEAN_KERNEL, sha[:39] + "4"),
         MockPolarisClient(challenger_speedup=1.5, challenger_delegation="cublasSgemm via popped trap"))

    print("\n" + "=" * 72)
    print("Demo complete. Real wiring TODO: Polaris API + GitHub bot token + SN74 identity (see README).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
