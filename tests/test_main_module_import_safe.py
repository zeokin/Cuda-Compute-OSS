"""Package ``__main__`` modules must be import-safe."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["matmul.__main__", "strategy.__main__"])
def test_importing_main_module_does_not_execute_cli(module):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
