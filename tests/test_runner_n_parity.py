"""matmul and strategy must agree on the runner `n` boundary.

The two packages are deliberately standalone copies of the same orchestration,
so a boundary check added to one silently drifts out of the other. `matmul`
rejected a degenerate n up front while `strategy` accepted it and only failed
later -- after a Backend (a real GPU) had already been demanded, and then with
a raw ZeroDivisionError from storage's `// (n * 8)` block-size math.

These are parity tests: each public runner entry point is exercised through the
SAME table, so the two packages cannot drift apart again. A stub Backend that
raises on construction proves the check runs *before* any device work, which is
what lets this run on CPU with no GPU/torch.
"""
from __future__ import annotations

import pytest

import matmul.runner as matmul_runner
import strategy.runner as strategy_runner
from matmul.config import Config as MatmulConfig
from strategy.config import Config as StrategyConfig

# n values that are not a positive integer. `True` is included because bool is a
# subclass of int -- matmul.run(True) must not be read as n=1.
BAD_N = (0, -1, 1.5, True)

# (label, module, callable-name, config-factory) for every runner entry point.
ENTRY_POINTS = [
    ("matmul.run", matmul_runner, "run", lambda: MatmulConfig(verbose=False)),
    ("strategy.run", strategy_runner, "run", lambda: StrategyConfig(verbose=False)),
    ("strategy.compare", strategy_runner, "compare", lambda: StrategyConfig(verbose=False)),
]


class _ExplodingBackend:
    """Stands in for Backend so any device work fails loudly instead of being
    skipped for lack of a GPU."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("n must be validated before Backend construction")


@pytest.mark.parametrize("label,module,fn_name,make_cfg", ENTRY_POINTS,
                         ids=[e[0] for e in ENTRY_POINTS])
@pytest.mark.parametrize("n", BAD_N, ids=[repr(v) for v in BAD_N])
def test_runner_rejects_bad_n_before_backend(monkeypatch, label, module, fn_name, make_cfg, n):
    monkeypatch.setattr(module, "Backend", _ExplodingBackend)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        getattr(module, fn_name)(n, make_cfg())


@pytest.mark.parametrize("label,module,fn_name,make_cfg", ENTRY_POINTS,
                         ids=[e[0] for e in ENTRY_POINTS])
def test_runner_accepts_smallest_valid_n(monkeypatch, label, module, fn_name, make_cfg):
    """n=1 is valid, so it must get past the check and reach Backend (our stub),
    guarding the boundary against an off-by-one."""
    monkeypatch.setattr(module, "Backend", _ExplodingBackend)
    with pytest.raises(AssertionError, match="before Backend construction"):
        getattr(module, fn_name)(1, make_cfg())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
