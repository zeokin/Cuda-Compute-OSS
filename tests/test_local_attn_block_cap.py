"""local_window_attention auto query-block must not track window (#317)."""
from __future__ import annotations

import ast
import os

import pytest

from attention import hybrid


def test_default_block_constant() -> None:
    assert hybrid._DEFAULT_LOCAL_QUERY_BLOCK == 64


def test_auto_block_formula_ignores_window() -> None:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "attention", "hybrid.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "local_window_attention")
    block_assigns = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "block" for t in n.targets)
    ]
    assert block_assigns
    src = ast.dump(block_assigns[0].value)
    assert "window" not in src
    assert "_DEFAULT_LOCAL_QUERY_BLOCK" in src
