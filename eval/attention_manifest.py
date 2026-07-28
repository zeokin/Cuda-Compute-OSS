"""Versioned workload manifest for the attention-foundation evaluator."""
from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmarks" / "attention-foundation-v1-rtx5070ti.json"
VALID_STATUSES = frozenset({"draft", "frozen", "retired"})
VALID_MODES = frozenset({"prefill", "decode", "guard"})


@dataclass(frozen=True)
class AttentionWorkload:
    id: str
    mode: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    dim: int
    dtype: str
    causal: bool
    scored: bool
    oracle: bool
    layout: str


@dataclass(frozen=True)
class AttentionManifest:
    path: Path
    sha256: str
    raw: dict
    workloads: tuple[AttentionWorkload, ...]

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def status(self) -> str:
        return self.raw["status"]

    @property
    def merge_enabled(self) -> bool:
        return bool(self.raw["decision"].get("merge_enabled"))

    @property
    def is_active(self) -> bool:
        decision = self.raw["decision"]
        return (
            self.status == "frozen"
            and self.merge_enabled
            and not decision.get("calibration_required", True)
        )


def _positive_int(raw: dict, name: str) -> int:
    value = raw.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"workload {name} must be a positive integer")
    return value


def _validate_workload(raw: dict) -> AttentionWorkload:
    required = {
        "id", "mode", "batch", "heads", "q_len", "kv_len", "dim",
        "dtype", "causal", "scored", "oracle",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"workload is missing fields: {sorted(missing)}")
    if not raw["id"] or not isinstance(raw["id"], str):
        raise ValueError("workload id must be a non-empty string")
    if raw["mode"] not in VALID_MODES:
        raise ValueError(f"unknown workload mode: {raw['mode']!r}")
    if raw["dtype"] not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"unsupported workload dtype: {raw['dtype']!r}")
    layout = raw.get("layout", "contiguous")
    if layout not in {"contiguous", "noncontiguous"}:
        raise ValueError(f"unsupported workload layout: {layout!r}")
    for name in ("causal", "scored", "oracle"):
        if not isinstance(raw[name], bool):
            raise ValueError(f"workload {name} must be boolean")
    if raw["mode"] == "decode" and raw["q_len"] != 1:
        raise ValueError("decode workloads must use q_len=1 in benchmark v1")
    return AttentionWorkload(
        id=raw["id"],
        mode=raw["mode"],
        batch=_positive_int(raw, "batch"),
        heads=_positive_int(raw, "heads"),
        q_len=_positive_int(raw, "q_len"),
        kv_len=_positive_int(raw, "kv_len"),
        dim=_positive_int(raw, "dim"),
        dtype=raw["dtype"],
        causal=raw["causal"],
        scored=raw["scored"],
        oracle=raw["oracle"],
        layout=layout,
    )


def validate_manifest(raw: dict) -> tuple[AttentionWorkload, ...]:
    if raw.get("schema_version") != 1:
        raise ValueError("attention manifest schema_version must be 1")
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise ValueError("attention manifest id must be a non-empty string")
    if raw.get("status") not in VALID_STATUSES:
        raise ValueError(f"manifest status must be one of {sorted(VALID_STATUSES)}")
    if not isinstance(raw.get("candidate"), str) or ":" not in raw["candidate"]:
        raise ValueError("candidate must use module:callable syntax")
    for section in ("era", "measurement", "correctness", "decision"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"manifest section {section!r} must be an object")
    measurement = raw["measurement"]
    for name in ("warmups", "repetitions"):
        if isinstance(measurement.get(name), bool) or not isinstance(measurement.get(name), int):
            raise ValueError(f"measurement.{name} must be an integer")
        if measurement[name] <= 0:
            raise ValueError(f"measurement.{name} must be > 0")
    workloads = tuple(_validate_workload(item) for item in raw.get("workloads", []))
    if not workloads:
        raise ValueError("attention manifest must define workloads")
    ids = [workload.id for workload in workloads]
    if len(ids) != len(set(ids)):
        raise ValueError("attention workload ids must be unique")
    if not any(workload.scored for workload in workloads):
        raise ValueError("attention manifest must contain a scored workload")
    if raw["status"] == "frozen":
        decision = raw["decision"]
        needed = (
            "minimum_speedup_percent",
            "maximum_per_workload_regression_percent",
            "maximum_vram_regression_percent",
        )
        if decision.get("calibration_required"):
            raise ValueError("a frozen manifest cannot require calibration")
        if any(decision.get(name) is None for name in needed):
            raise ValueError("a frozen manifest must contain calibrated decision thresholds")
        tiers = decision.get("tiers_percent")
        if not isinstance(tiers, dict) or set(tiers) != {"S", "M", "L"}:
            raise ValueError("a frozen manifest must define S/M/L tier thresholds")
        if not all(isinstance(value, (int, float)) and value >= 0 for value in tiers.values()):
            raise ValueError("tier thresholds must be non-negative numbers")
        if not tiers["S"] <= tiers["M"] <= tiers["L"]:
            raise ValueError("tier thresholds must be monotonic S <= M <= L")
    return workloads


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> AttentionManifest:
    manifest_path = Path(path).resolve()
    data = manifest_path.read_bytes()
    raw = json.loads(data.decode("utf-8"))
    workloads = validate_manifest(raw)
    return AttentionManifest(
        path=manifest_path,
        sha256=hashlib.sha256(data).hexdigest(),
        raw=raw,
        workloads=workloads,
    )


def load_candidate(spec: str) -> Callable:
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    candidate = getattr(module, attr, None)
    if not callable(candidate):
        raise ValueError(f"candidate {spec!r} is not callable")
    return candidate


__all__ = [
    "AttentionManifest",
    "AttentionWorkload",
    "DEFAULT_MANIFEST",
    "load_candidate",
    "load_manifest",
    "validate_manifest",
]
