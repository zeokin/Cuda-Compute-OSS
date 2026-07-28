"""Keep the public README, landing page, template, and manifest consistent."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "attention-foundation-v1-rtx5070ti.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_public_surfaces_name_the_protected_benchmark():
    benchmark = _manifest()["id"]
    for path in (
        ROOT / "README.md",
        ROOT / "index" / "index.html",
        ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
    ):
        assert benchmark in path.read_text(encoding="utf-8"), path


def test_public_status_matches_manifest_activation_lock():
    manifest = _manifest()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "index" / "index.html").read_text(encoding="utf-8")
    active = (
        manifest["status"] == "frozen"
        and manifest["decision"]["calibration_required"] is False
        and manifest["decision"]["merge_enabled"] is True
    )
    if active:
        assert "ACTIVE" in readme
        assert "ACTIVE" in page
    else:
        assert "DRAFT — miner PRs closed" in readme
        assert "DRAFT · MINER PRS CLOSED" in page


def test_manifest_contains_the_declared_nine_case_basket():
    workloads = _manifest()["workloads"]
    assert len(workloads) == 9
    assert sum(item["scored"] for item in workloads) == 7
    assert [item["q_len"] for item in workloads if item["mode"] == "prefill"] == [
        1024, 2048, 4096, 8192,
    ]
    assert [item["kv_len"] for item in workloads if item["mode"] == "decode"] == [
        1024, 4096, 8192,
    ]


def test_removed_public_roadmap_is_not_linked():
    public_files = [
        ROOT / "README.md",
        ROOT / "index" / "index.html",
        ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
    ]
    forbidden = ("docs/roadmap.md", "docs/evaluation.md", "CONTRIBUTING.md", "BENCHMARKS.md")
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        assert not any(item in text for item in forbidden), path
