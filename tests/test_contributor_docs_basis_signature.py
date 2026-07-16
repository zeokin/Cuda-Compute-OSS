"""Keep every contributor-facing basis() snippet on the real interface.

`multiply_subspace` forwards the VRAM budget only to a basis() that declares
`frac` (an inspect.signature gate whose `else` is a legacy-compat shim), so any
snippet a contributor copies must declare it -- otherwise their transform lands
in the legacy branch and silently streams its basis at the default fraction
instead of Config.vram_fraction (the bug #211 fixed for rsvd).

There are three copies a contributor actually reads, and they must all agree
with `Transform.basis`:

  * CONTRIBUTING.md -- "What you actually change", the canonical
    "that is enough to be scored" snippet.
  * strategy/README.md -- "Register your own (the updatable hook)".
  * strategy/examples/run_example.py -- the worked FirstAxes transform.

(strategy/examples/transform_template.py is covered by its own test.)

The markdown snippets are regex-checked and the example is AST-parsed rather
than imported: run_example.py runs a real subspace_matmul at module scope, which
needs a GPU. Pure parsing; no GPU needed.

Run:  python tests/test_contributor_docs_basis_signature.py
"""
import ast
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.transforms import Transform

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MARKDOWN = {
    "CONTRIBUTING.md": os.path.join(_ROOT, "CONTRIBUTING.md"),
    "strategy/README.md": os.path.join(_ROOT, "strategy", "README.md"),
}
_RUN_EXAMPLE = os.path.join(_ROOT, "strategy", "examples", "run_example.py")


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


def test_markdown_snippets_declare_frac():
    """The snippets contributors copy out of the docs are the contract."""
    for label, path in _MARKDOWN.items():
        sigs = re.findall(r"def basis\(([^)]*)\)", _read(path))
        assert sigs, f"no `def basis(...)` snippet found in {label}"
        for sig in sigs:
            assert "frac" in sig, f"{label} basis() snippet omits frac: def basis({sig})"


def test_run_example_custom_transform_declares_frac():
    defs = _basis_defs(_read(_RUN_EXAMPLE))
    assert defs, "no basis() definition found in strategy/examples/run_example.py"
    for params in defs:
        assert "frac" in params, f"run_example basis() omits frac: {params}"


def test_run_example_basis_matches_the_base_class_contract():
    for params in _basis_defs(_read(_RUN_EXAMPLE)):
        assert params == _base_params()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
