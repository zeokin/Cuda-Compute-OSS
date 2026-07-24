"""#258: miner-legal path — strategy raises RuntimeError so eval/cli exits 2.

eval/cli.py is CODEOWNERS-protected and only catches RuntimeError. Unknown
--transforms used to raise KeyError; bad --rank-m used to raise ValueError.
Both escaped as raw tracebacks. Raising RuntimeError from strategy closes the
gap without editing eval/.
"""
from __future__ import annotations

import pytest

from strategy.subspace import validate_rank_m
from strategy.transforms import get_transform


def test_unknown_transform_is_runtime_error():
    with pytest.raises(RuntimeError, match="unknown transform"):
        get_transform("not-a-real-transform")


@pytest.mark.parametrize("bad", (0, -1, 10_000))
def test_bad_rank_m_is_runtime_error(bad):
    with pytest.raises(RuntimeError, match=r"rank_m must be in"):
        validate_rank_m(bad, n=64)


def test_eval_cli_maps_strategy_runtime_errors_to_exit_2(monkeypatch, capsys):
    from eval import cli as eval_cli

    def boom(_ev):
        raise RuntimeError("unknown transform 'not-a-real-transform'")

    monkeypatch.setattr(eval_cli, "evaluate", boom)
    assert eval_cli.main(["--n", "8", "--transforms", "not-a-real-transform"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err
