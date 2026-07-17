"""Keep every contributor-facing basis() snippet on the real interface.

`multiply_subspace` forwards the VRAM budget only to a basis() that declares
`frac` (an inspect.signature gate whose `else` is a legacy-compat shim), so any
snippet a contributor copies must declare it -- otherwise their transform lands
in the legacy branch and silently streams its basis at the default fraction
instead of Config.vram_fraction (the bug #211 fixed for rsvd).

The template was brought in line already; these tests cover every other copy a
contributor actually reads:

  * CONTRIBUTING.md's "What you actually change" snippet -- the canonical
    "that is enough to be scored" example.
  * strategy/README.md's "Register your own (the updatable hook)" -- the copy a
    contributor meets first, and the one this file used to leave uncovered, which
    is how it stayed on the old signature (#274).
  * strategy/examples/run_example.py's FirstAxes -- the worked custom transform.

All are parsed as text/AST rather than imported: run_example.py runs a real
subspace_matmul at module scope, which needs a GPU. Pure parsing; no GPU needed.

Run:  python tests/test_contributor_docs_basis_signature.py
"""
import ast
import inspect
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.transforms import Transform

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONTRIBUTING = os.path.join(_ROOT, "CONTRIBUTING.md")
_README = os.path.join(_ROOT, "strategy", "README.md")
_RUN_EXAMPLE = os.path.join(_ROOT, "strategy", "examples", "run_example.py")

# Every prose copy of the hook. Driven through one table so a new doc cannot be
# added -- or an existing one silently rolled back -- without being checked.
_MARKDOWN_SNIPPETS = ((_CONTRIBUTING, "CONTRIBUTING.md"), (_README, "strategy/README.md"))


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _base_params() -> list:
    return list(inspect.signature(Transform.basis).parameters)


def _basis_defs(source: str) -> list:
    """Every `def basis(...)` parameter list found by parsing the module."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "basis":
            args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
            out.append(args)
    return out


@pytest.mark.parametrize("path,label", _MARKDOWN_SNIPPETS, ids=[s[1] for s in _MARKDOWN_SNIPPETS])
def test_markdown_snippet_declares_frac(path, label):
    """Every prose basis() a contributor copies must show the real signature."""
    snippets = re.findall(r"def basis\(([^)]*)\)", _read(path))
    assert snippets, f"no `def basis(...)` snippet found in {label}"
    for sig in snippets:
        assert "frac" in sig, f"{label} basis() snippet omits frac: def basis({sig})"


@pytest.mark.parametrize("path,label", _MARKDOWN_SNIPPETS, ids=[s[1] for s in _MARKDOWN_SNIPPETS])
def test_markdown_snippet_matches_the_base_class_contract(path, label):
    """Not just frac: the whole parameter list must match Transform.basis, so a
    snippet cannot drift on any other argument either. Defaults are stripped --
    the snippets legitimately show `A=None`, the contract is the parameter names
    and their order."""
    for sig in re.findall(r"def basis\(([^)]*)\)", _read(path)):
        got = [p.strip().split("=")[0].strip() for p in sig.split(",")]
        assert got == _base_params(), (
            f"{label}: def basis({', '.join(got)}) does not match "
            f"Transform.basis({', '.join(_base_params())})"
        )


def test_run_example_custom_transform_declares_frac():
    """The worked example's custom transform must show the real signature too."""
    defs = _basis_defs(_read(_RUN_EXAMPLE))
    assert defs, "no basis() definition found in strategy/examples/run_example.py"
    for params in defs:
        assert "frac" in params, f"run_example basis() omits frac: {params}"


def test_run_example_basis_matches_the_base_class_contract():
    for params in _basis_defs(_read(_RUN_EXAMPLE)):
        assert params == _base_params()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
