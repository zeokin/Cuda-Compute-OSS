"""Importing a package's ``__main__`` must not execute its CLI.

``python -m matmul`` / ``-m strategy`` / ``-m attention.benchmark`` run the CLI
because that sets ``__name__ == "__main__"``. But a plain
``import matmul.__main__`` (done by test collectors, coverage tools, doc
importers, ``runpy``) must be a no-op. Without the ``if __name__ == "__main__"``
guard, ``raise SystemExit(main())`` runs at *import* time -- main() parses argv
and, on a GPU-less box, fails building the Backend and raises ``SystemExit(2)``,
so the import aborts. attention/__main__.py already guards this; these pin the
same contract for all three entry points.

Fresh subprocess per module so the import genuinely re-runs (an in-process
import would hit the module cache and prove nothing).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize(
    "module", ["matmul.__main__", "strategy.__main__", "attention.__main__"]
)
def test_importing_main_module_is_side_effect_free(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing {module} exited {result.returncode} -- it ran its CLI at "
        f"import time (missing `if __name__ == \"__main__\"` guard). "
        f"stderr: {result.stderr.strip()}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
