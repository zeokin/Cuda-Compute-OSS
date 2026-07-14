"""CPU-only tests for matmul runner disk cleanup (--keep flag), including that
cleanup still runs when the multiply itself raises -- run() only removed its
memmaps at the very end, so an exception anywhere above that (a bad --device
index, a mid-multiply OOM) skipped it and leaked the disk-backed A/B/C files.

Mirrors strategy/tests/test_runner_keep.py for the sibling matmul runner.

Run:  python tests/test_matmul_runner_cleanup.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul.config import Config
from matmul import runner


class _FakeBackend:
    name = "fake"

    def synchronize(self) -> None:
        pass

    def host_available_bytes(self) -> int:
        return 10**12


def _patch_disk_runner(monkeypatch):
    """Force disk-backed run() without touching a real GPU."""
    n = 4
    dt = np.float32

    def _touch(path):
        if path:
            with open(path, "wb") as fh:
                fh.write(b"\0")

    def fake_generate(n_, dt_, on_disk, path, *args, **kwargs):
        if on_disk and path:
            _touch(path)
        return np.zeros((n_, n_), dtype=dt_)

    def fake_allocate(n_, dt_, on_disk, path):
        if on_disk and path:
            _touch(path)
            return np.memmap(path, mode="r+", dtype=dt_, shape=(n_, n_))
        return np.zeros((n_, n_), dtype=dt_)

    monkeypatch.setattr(runner.storage, "should_use_disk", lambda *a, **k: True)
    monkeypatch.setattr(runner.storage, "generate", fake_generate)
    monkeypatch.setattr(runner.storage, "allocate", fake_allocate)
    monkeypatch.setattr(runner, "Backend", lambda *a, **k: _FakeBackend())
    monkeypatch.setattr(
        runner.gemm,
        "multiply",
        lambda *a, **k: {"mode": "exact"},
    )
    return n, dt


def _memmap_paths(workdir: str):
    return [os.path.join(workdir, name) for name in ("A.dat", "B.dat", "C.dat")]


def _raise(*_a, **_k):
    raise RuntimeError("boom")


def test_run_keep_true_preserves_memmaps(tmp_path, monkeypatch):
    removed: list[str] = []
    monkeypatch.setattr(os, "remove", lambda p: removed.append(p))
    n, _ = _patch_disk_runner(monkeypatch)
    cfg = Config(workdir=str(tmp_path), storage="disk", verbose=False)
    runner.run(n, cfg, keep=True)
    assert removed == []


def test_run_keep_false_removes_memmaps(tmp_path, monkeypatch):
    removed: list[str] = []
    monkeypatch.setattr(os, "remove", lambda p: removed.append(p))
    n, _ = _patch_disk_runner(monkeypatch)
    cfg = Config(workdir=str(tmp_path), storage="disk", verbose=False)
    runner.run(n, cfg, keep=False)
    assert set(removed) == set(_memmap_paths(str(tmp_path)))


def test_run_removes_memmaps_when_multiply_raises(tmp_path, monkeypatch):
    removed: list[str] = []
    monkeypatch.setattr(os, "remove", lambda p: removed.append(p))
    n, _ = _patch_disk_runner(monkeypatch)
    monkeypatch.setattr(runner.gemm, "multiply", _raise)
    cfg = Config(workdir=str(tmp_path), storage="disk", verbose=False)
    try:
        runner.run(n, cfg, keep=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError to propagate")
    assert set(removed) == set(_memmap_paths(str(tmp_path)))


def test_run_keeps_memmaps_on_raise_when_keep_true(tmp_path, monkeypatch):
    removed: list[str] = []
    monkeypatch.setattr(os, "remove", lambda p: removed.append(p))
    n, _ = _patch_disk_runner(monkeypatch)
    monkeypatch.setattr(runner.gemm, "multiply", _raise)
    cfg = Config(workdir=str(tmp_path), storage="disk", verbose=False)
    try:
        runner.run(n, cfg, keep=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError to propagate")
    assert removed == []


if __name__ == "__main__":
    try:
        import pytest
    except ImportError:
        print("SKIP  tests/test_matmul_runner_cleanup.py (pytest required)")
        sys.exit(0)

    raise SystemExit(pytest.main([__file__, "-v"]))
