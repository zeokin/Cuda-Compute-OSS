"""CPU-only tests for the --spectral-alpha decay-rate knob (issue #231).

`--fill decaying-spectrum` builds a rank-r matrix whose component k has weight
k**-alpha. storage.generate/_fill_decaying_spectrum accept `spectral_alpha`, but
the strategy CLI and runner.run/compare never forwarded it, so from the CLI the
decay rate was stuck at the default 1.0. These tests pin the whole plumbing:
the CLI flag exists and is validated, runner.run/compare accept it, and alpha
actually controls the spectral decay. Pure NumPy; no GPU needed.

Run:  python tests/test_spectral_alpha.py
"""
import inspect
import io
import os
import sys
from contextlib import redirect_stderr

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import storage, runner
from strategy.cli import build_parser, main


def test_cli_exposes_spectral_alpha_with_default_one():
    args = build_parser().parse_args([])
    assert args.spectral_alpha == 1.0
    args = build_parser().parse_args(["--spectral-alpha", "2.5"])
    assert args.spectral_alpha == 2.5


def test_runner_run_and_compare_accept_spectral_alpha():
    assert "spectral_alpha" in inspect.signature(runner.run).parameters
    assert "spectral_alpha" in inspect.signature(runner.compare).parameters


def test_cli_rejects_negative_spectral_alpha_cleanly():
    # Validated before the GPU backend is built, so the message is about the flag
    # and the exit code is a clean 2 (not a traceback), even with no GPU present.
    err = io.StringIO()
    with redirect_stderr(err):
        rc = main(["--spectral-alpha", "-0.5", "--quiet"])
    assert rc == 2
    assert "spectral-alpha" in err.getvalue()


def _top_energy_fraction(mat, k=1):
    # Fraction of the total spectral energy carried by the top-k singular values.
    sv = np.linalg.svd(np.asarray(mat, dtype=np.float64), compute_uv=False)
    return float((sv[:k] ** 2).sum() / (sv ** 2).sum())


def test_spectral_alpha_controls_decay():
    n, seed = 64, 0
    flat = storage.generate(n, np.float32, False, None, seed, "decaying-spectrum",
                            data_rank=32, spectral_alpha=0.0)
    steep = storage.generate(n, np.float32, False, None, seed, "decaying-spectrum",
                             data_rank=32, spectral_alpha=3.0)
    # A larger alpha decays faster, so more energy concentrates in the top modes.
    assert _top_energy_fraction(steep, k=4) > _top_energy_fraction(flat, k=4)
    # And the two spectra are genuinely different (alpha is not ignored).
    assert not np.allclose(np.asarray(flat), np.asarray(steep))


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
