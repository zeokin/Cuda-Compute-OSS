"""`strategy --compare --quiet` must still print a one-line summary.

Previously main() ran runner.compare() (which only prints when verbose, i.e.
NOT under --quiet), discarded its result, and returned before the quiet-summary
block -- so `--compare --quiet` produced no output at all.

runner.compare() is stubbed so this runs on CPU with no GPU/torch (the flag
parsing and summary formatting are what we exercise).
"""
from __future__ import annotations

import pytest

from strategy import cli as strategy_cli


def test_compare_quiet_prints_summary(monkeypatch, capsys):
    fake = {
        "n": 8, "exact_seconds": 0.100, "smart_seconds": 0.050, "speedup": 2.0,
        "rel_err": 1.2e-3, "flop_ratio": 4.0,
        "exact_mode": "exact(streamed)", "smart_mode": "subspace(M=8)",
    }
    monkeypatch.setattr(strategy_cli.runner, "compare", lambda *a, **k: fake)

    rc = strategy_cli.main(["--compare", "--quiet", "--n", "8"])

    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip(), "--compare --quiet printed nothing"
    assert "speedup 2.00x" in out
    assert "rel_err 1.20e-03" in out


def test_compare_nonquiet_does_not_double_print(monkeypatch, capsys):
    # When NOT quiet, compare() itself prints; main() must not add the extra
    # quiet summary on top.
    monkeypatch.setattr(strategy_cli.runner, "compare", lambda *a, **k: {
        "exact_seconds": 0.1, "smart_seconds": 0.05, "speedup": 2.0, "rel_err": 1e-3,
    })
    rc = strategy_cli.main(["--compare", "--n", "8"])
    assert rc == 0
    assert "speedup 2.00x" not in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
