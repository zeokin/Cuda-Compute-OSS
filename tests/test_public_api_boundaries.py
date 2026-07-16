"""CPU-only regression tests for public API validation boundaries."""
from __future__ import annotations

import numpy as np
import pytest

import matmul
import matmul.runner as matmul_runner
from matmul.config import Config


class _NoBackend:
    def __init__(self, *args, **kwargs):
        raise AssertionError("validation must run before backend construction")


def test_matmul_rejects_output_that_aliases_an_input(monkeypatch):
    monkeypatch.setattr(matmul, "Backend", _NoBackend)
    A = np.eye(4, dtype=np.float32)
    B = np.eye(4, dtype=np.float32)

    for out in (A, B, A[:, :]):
        with pytest.raises(ValueError, match="out must not share memory with A or B"):
            matmul.matmul(A, B, out=out, config=Config(dtype="fp32", verbose=False))


@pytest.mark.parametrize("name", ("accumulate_fp32", "force_tiled"))
@pytest.mark.parametrize("value", ("false", 0, 1, None))
def test_matmul_config_rejects_non_boolean_compute_flags(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a bool"):
        Config(**{name: value})


@pytest.mark.parametrize("n", (0, -1, 1.5, True))
def test_matmul_runner_rejects_invalid_n_before_backend(monkeypatch, n):
    monkeypatch.setattr(matmul_runner, "Backend", _NoBackend)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        matmul_runner.run(n, Config(verbose=False))


@pytest.mark.parametrize("fn_name", ("run", "compare"))
@pytest.mark.parametrize("n", (0, -1, 1.5, True))
def test_strategy_runner_rejects_invalid_n_before_backend(monkeypatch, fn_name, n):
    import strategy.runner as strategy_runner
    from strategy.config import Config as StrategyConfig

    monkeypatch.setattr(strategy_runner, "Backend", _NoBackend)
    fn = getattr(strategy_runner, fn_name)
    with pytest.raises(ValueError, match="n must be a positive integer"):
        fn(n, StrategyConfig(verbose=False))


@pytest.mark.parametrize("value", ("false", 0, None))
def test_attention_rejects_non_boolean_causal(value):
    torch = pytest.importorskip("torch")
    from attention.hybrid import local_window_attention
    from attention.reference import exact_attention

    q = k = v = torch.randn(1, 1, 4, 2)
    with pytest.raises(ValueError, match="causal must be a bool"):
        local_window_attention(q, k, v, window=1, causal=value)
    with pytest.raises(ValueError, match="causal must be a bool"):
        exact_attention(q, k, v, causal=value)
