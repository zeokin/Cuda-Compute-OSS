"""Invalid CLI arguments must be reported cleanly (exit code 2 + an ``error:``
line on stderr), never surfaced as an uncaught traceback.

Both ``matmul`` and ``strategy`` validate ``--n`` and build/validate their
``Config`` *before* any device work, so every case here is rejected on CPU with
no GPU/PyTorch present -- which is exactly how they run in PR CI.
"""
from __future__ import annotations

import pytest

from matmul import cli as matmul_cli
from strategy import cli as strategy_cli

BAD_ARGS = [
    ["--vram-fraction", "1.5", "--n", "8"],   # vram_fraction > 0.95
    ["--vram-fraction", "0", "--n", "8"],     # vram_fraction <= 0
    ["--n", "0"],                             # non-positive n
    ["--n", "-4"],                            # negative n
]


@pytest.mark.parametrize("main", [matmul_cli.main, strategy_cli.main],
                         ids=["matmul", "strategy"])
@pytest.mark.parametrize("argv", BAD_ARGS, ids=lambda a: " ".join(a))
def test_bad_args_exit_cleanly(main, argv, capsys):
    rc = main(argv)
    assert rc == 2, f"expected exit 2 for {argv}, got {rc}"
    assert "error:" in capsys.readouterr().err


# --data-rank is strategy-only (matmul has no such flag), so it can't share
# BAD_ARGS above, which is parametrized across both CLIs.
STRATEGY_BAD_DATA_RANK_ARGS = [
    ["--n", "8", "--data-rank", "0"],    # non-positive: a rank-0 "benchmark"
                                          # isn't a meaningful input, same as --n 0
    ["--n", "8", "--data-rank", "-3"],   # negative: previously an uncaught
                                          # traceback from _fill_lowrank's
                                          # 1/sqrt(rank) and negative-size rng draw
]


@pytest.mark.parametrize("argv", STRATEGY_BAD_DATA_RANK_ARGS, ids=lambda a: " ".join(a))
def test_bad_data_rank_exits_cleanly(argv, capsys):
    rc = strategy_cli.main(argv)
    assert rc == 2, f"expected exit 2 for {argv}, got {rc}"
    assert "error:" in capsys.readouterr().err


STRATEGY_BAD_SPECTRAL_ALPHA_ARGS = [
    ["--n", "8", "--spectral-alpha", "-0.5"],
    ["--n", "8", "--spectral-alpha", "nan"],
    ["--n", "8", "--spectral-alpha", "inf"],
    ["--n", "8", "--spectral-alpha=-inf"],
]


@pytest.mark.parametrize("argv", STRATEGY_BAD_SPECTRAL_ALPHA_ARGS,
                         ids=lambda a: " ".join(a))
def test_bad_spectral_alpha_exits_cleanly(argv, capsys):
    rc = strategy_cli.main(argv)
    assert rc == 2, f"expected exit 2 for {argv}, got {rc}"
    assert "error:" in capsys.readouterr().err


STRATEGY_BAD_RANK_M_ARGS = [
    ["--n", "8", "--rank-m", "0"],     # non-positive: rank_m must be >= 1
    ["--n", "8", "--rank-m", "-2"],    # negative: previously hit GPU before subspace
    ["--n", "8", "--rank-m", "100"],   # rank_m > n: previously hit GPU before subspace
]


@pytest.mark.parametrize("argv", STRATEGY_BAD_RANK_M_ARGS, ids=lambda a: " ".join(a))
def test_bad_rank_m_exits_cleanly(argv, capsys):
    rc = strategy_cli.main(argv)
    assert rc == 2, f"expected exit 2 for {argv}, got {rc}"
    assert "error:" in capsys.readouterr().err


def test_positive_rank_m_is_unaffected(capsys):
    # --rank-m 1 (smallest valid rank) must not be rejected by validation.
    rc = strategy_cli.main(["--n", "8", "--rank-m", "1", "--quiet"])
    if rc == 2:
        assert "--rank-m" not in capsys.readouterr().err


def test_positive_data_rank_is_unaffected(capsys):
    # --data-rank 1 (smallest valid rank) must not be rejected by validation --
    # this guards against the check being off-by-one. This test runs without a
    # GPU, so a rc==2 here may legitimately come from the "no GPU" path (this
    # module computes on GPU only); it must never come from a --data-rank
    # complaint.
    rc = strategy_cli.main(["--n", "8", "--data-rank", "1", "--quiet"])
    if rc == 2:
        assert "--data-rank" not in capsys.readouterr().err


# attention.benchmark is the third CLI; it must convert the same class of
# invalid-knob failures into exit 2 + an ``error:`` line, not an uncaught
# traceback (#201). These are benchmark-only knobs (temperature, landmarks,
# branch weights, landmark-policy) that AttentionSpec / the hybrid helpers
# reject, so none is shared with BAD_ARGS above. run_once validates before any
# device work, and its _torch()/GPU path also raises cleanly here -- so with or
# without PyTorch installed, each case must return 2 rather than raise.
BENCHMARK_BAD_ARGS = [
    ["--batch", "0", "--seq", "8"],            # AttentionSpec: batch > 0
    ["--heads", "0", "--seq", "8"],            # AttentionSpec: heads > 0
    ["--seq", "0"],                            # AttentionSpec: seq > 0
    ["--seq", "8", "--local-weight", "-1"],    # AttentionSpec: weights >= 0
    ["--seq", "8", "--local-weight", "0", "--global-weight", "0"],  # one must be > 0
    # temperature / landmarks are only consumed by the modes that use them, so
    # pair each with that mode -- otherwise the invalid value is never reached.
    ["--seq", "8", "--mode", "corrfft", "--temperature", "0"],   # temperature > 0
    ["--seq", "8", "--mode", "landmark", "--landmarks", "0"],    # num_landmarks > 0
    ["--seq", "8", "--mode", "adaptive", "--gate-strength", "-0.1"],  # gate_strength >= 0
]


@pytest.mark.parametrize("argv", BENCHMARK_BAD_ARGS, ids=lambda a: " ".join(a))
def test_attention_benchmark_bad_args_exit_cleanly(argv, capsys):
    from attention import benchmark as attention_benchmark

    rc = attention_benchmark.main(argv)
    assert rc == 2, f"expected exit 2 for {argv}, got {rc}"
    assert "error:" in capsys.readouterr().err


def test_strategy_unknown_transform_exits_cleanly(capsys):
    rc = strategy_cli.main(["--n", "8", "--transform", "nope", "--quiet"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "unknown transform" in err



def test_get_transform_unknown_raises_runtime_error():
    """Unknown names must be RuntimeError so eval/cli's except RuntimeError
    covers them without editing maintainer-owned eval/ (#258 / #199 gap)."""
    from strategy.transforms import get_transform

    with pytest.raises(RuntimeError, match="unknown transform"):
        get_transform("bogus")


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("unknown transform 'bogus'"),
        RuntimeError("rank_m must be in [1, n]; got 0 for n=8"),
    ],
    ids=["unknown-transform", "bad-rank-m"],
)
def test_eval_cli_runtime_user_errors_exit_cleanly(monkeypatch, capsys, exc):
    """eval/cli already catches RuntimeError; strategy now raises that family
    for the #258 user-input cases miners can fix without touching eval/."""
    from eval import cli as eval_cli

    monkeypatch.setattr(eval_cli, "evaluate", lambda _ev: (_ for _ in ()).throw(exc))
    rc = eval_cli.main(["--n", "8", "--transforms", "bogus"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
