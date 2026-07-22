"""local_window_attention auto query-block must not track window (#317).

Default ``block = max(64, window)`` made wide-window runs allocate near-full
score tensors and OOM. The automatic block is now a fixed cap; explicit
``block_size`` is unchanged.

CPU-safe: the formula regression runs without torch. Numerical checks skip
cleanly when torch is absent (CI eval-policy pattern).

Run:  python tests/test_local_attn_block_cap.py
"""
from __future__ import annotations

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

import pytest

from attention import hybrid


def test_default_block_constant_does_not_track_window():
    assert hybrid._DEFAULT_LOCAL_QUERY_BLOCK == 64


def test_auto_block_formula_ignores_window():
    """Parse local_window_attention: default branch must not reference window."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "attention",
        "hybrid.py",
    )
    tree = ast.parse(open(path, encoding="utf-8").read())
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "local_window_attention":
            fn = node
            break
    assert fn is not None
    # Find the assignment to ``block`` and ensure the else/default side does
    # not name ``window`` (the old max(64, window) bug).
    block_assigns = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "block" for t in n.targets)
    ]
    assert block_assigns, "expected a block = ... assignment"
    src = ast.dump(block_assigns[0].value)
    assert "window" not in src, src
    assert "_DEFAULT_LOCAL_QUERY_BLOCK" in src, src


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_default_block_independent_of_wide_window(monkeypatch):
    heights = []
    real_matmul = torch.matmul

    def spy_matmul(a, b):
        if getattr(a, "ndim", 0) == 4 and getattr(b, "ndim", 0) == 4:
            if a.shape[-1] == b.shape[-2]:
                heights.append(int(a.shape[-2]))
        return real_matmul(a, b)

    monkeypatch.setattr(torch, "matmul", spy_matmul)
    q = torch.randn(1, 1, 128, 4)
    hybrid.local_window_attention(q, q, q, window=96)  # window > 64
    assert heights, "expected score matmuls"
    assert max(heights) == hybrid._DEFAULT_LOCAL_QUERY_BLOCK
    assert max(heights) < 96


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_explicit_block_size_still_honored(monkeypatch):
    heights = []
    real_matmul = torch.matmul

    def spy_matmul(a, b):
        if getattr(a, "ndim", 0) == 4 and getattr(b, "ndim", 0) == 4:
            if a.shape[-1] == b.shape[-2]:
                heights.append(int(a.shape[-2]))
        return real_matmul(a, b)

    monkeypatch.setattr(torch, "matmul", spy_matmul)
    q = torch.randn(1, 1, 48, 4)
    hybrid.local_window_attention(q, q, q, window=8, block_size=7)
    assert max(heights) == 7


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_blocking_does_not_change_local_result():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 40, 4)
    k = torch.randn(1, 2, 40, 4)
    v = torch.randn(1, 2, 40, 4)
    ref = hybrid.local_window_attention(q, k, v, window=5, block_size=40)
    blocked = hybrid.local_window_attention(q, k, v, window=5, block_size=None)
    assert torch.allclose(ref, blocked, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
