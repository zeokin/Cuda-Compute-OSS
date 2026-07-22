"""Tests for eval.gpu_batch's queue planning logic -- no GPU or gh calls."""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path

from eval.gpu_batch import (
    EvalSpec,
    QueueItem,
    eval_args,
    load_queue,
    mock_result,
    plan_item,
    run_item,
    select_batch,
    wrap_result,
)


def _queue_file(items):
    d = tempfile.TemporaryDirectory()
    path = os.path.join(d.name, "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"queue": items}, f)
    return d, path


def test_load_queue_orders_by_position():
    tmp, path = _queue_file([
        {"pr": 7, "title": "later", "author": "bob", "head_sha": "b" * 40, "position": 2},
        {"pr": 3, "title": "first", "author": "alice", "head_sha": "a" * 40, "position": 1},
    ])
    try:
        queue = load_queue(path)
        assert [item.pr for item in queue] == [3, 7]
    finally:
        tmp.cleanup()


def test_select_batch_limit():
    tmp, path = _queue_file([
        {"pr": 1, "title": "a", "author": "a", "head_sha": "1", "position": 1},
        {"pr": 2, "title": "b", "author": "b", "head_sha": "2", "position": 2},
    ])
    try:
        assert [item.pr for item in select_batch(load_queue(path), 1)] == [1]
        assert [item.pr for item in select_batch(load_queue(path), 0)] == [1, 2]
    finally:
        tmp.cleanup()


def test_eval_args_omit_seed_by_default():
    args = eval_args(EvalSpec(transforms="mine", rank_m=128))
    assert "--seed" not in args
    assert args[args.index("--transforms") + 1] == "mine"
    assert args[args.index("--rank-m") + 1] == "128"


def test_eval_args_include_seed_when_reproducing():
    args = eval_args(EvalSpec(seed=123))
    assert args[args.index("--seed") + 1] == "123"


def test_eval_args_accept_active_python_command():
    args = eval_args(EvalSpec(), python_cmd=["C:/cuda/python.exe"])
    assert args[:3] == ["C:/cuda/python.exe", "-m", "eval"]


def test_plan_contains_sha_check_and_json_output():
    tmp, path = _queue_file([
        {"pr": 4, "title": "mine", "author": "alice", "head_sha": "abcdef1234567890",
         "position": 1},
    ])
    try:
        item = load_queue(path)[0]
        commands = plan_item(
            item,
            repo="owner/repo",
            workdir="_work",
            results_dir="_results",
            spec=EvalSpec(transforms="mine"),
        )
        joined = "\n".join(commands)
        assert "gh pr checkout 4" in joined
        assert "abcdef1234567890" in joined
        assert "python -m eval" in joined
        assert "--json" in joined
        # The planned destination is absolute so it survives the preceding
        # ``cd`` into the PR worktree; only its separator changes by platform.
        assert "pr-4-abcdef123456.json" in joined
    finally:
        tmp.cleanup()


def test_active_python_plan_is_portable_and_does_not_create_uv_environment():
    tmp, path = _queue_file([
        {"pr": 4, "title": "mine", "author": "alice", "head_sha": "abcdef1234567890",
         "position": 1},
    ])
    try:
        commands = plan_item(
            load_queue(path)[0],
            repo="owner/repo",
            workdir="_work",
            results_dir="_results",
            spec=EvalSpec(),
            active_python=True,
        )
        joined = "\n".join(commands)
        assert "uv sync" not in joined
        assert "$(find" not in joined
        assert "compileall -q matmul strategy eval tests examples" in joined
        assert sys.executable in joined
    finally:
        tmp.cleanup()


def test_active_python_run_uses_supplied_interpreter_for_every_pr_command(monkeypatch, tmp_path):
    item = QueueItem(pr=4, title="mine", author="alice", head_sha="a" * 40,
                     track="full-rank", transform="mine")
    calls = []

    def fake_run(cmd, *, cwd=None, capture=False):
        calls.append(cmd)
        if cmd[:3] == ["gh", "repo", "clone"]:
            Path(cmd[-1]).mkdir(parents=True)
        if cmd[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=item.head_sha + "\n")
        if cmd[-3:] == ["-m", "eval", "--json"]:
            return SimpleNamespace(stdout='{"config": {}, "transforms": {}}')
        if "-m" in cmd and "eval" in cmd:
            return SimpleNamespace(stdout='{"config": {}, "transforms": {}}')
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("eval.gpu_batch._run", fake_run)
    monkeypatch.setattr("eval.gpu_batch._rebase_onto_main", lambda checkout: True)
    monkeypatch.setattr("eval.gpu_batch._transform_touched", lambda checkout, name: True)
    out = run_item(
        item,
        repo="owner/repo",
        workdir=tmp_path / "work",
        results_dir=tmp_path / "results",
        spec=EvalSpec(),
        active_python=True,
    )

    assert out.exists()
    python_calls = [cmd for cmd in calls if cmd and cmd[0] == sys.executable]
    assert python_calls
    assert not any(cmd and cmd[0] == "uv" for cmd in calls)
    assert any("compileall" in cmd for cmd in python_calls)
    assert any("CUDA is unavailable" in " ".join(cmd) for cmd in python_calls)


def test_clear_readonly_checkout_entry_retries_once(monkeypatch):
    from eval.gpu_batch import _clear_readonly_and_retry

    calls = []
    monkeypatch.setattr("eval.gpu_batch.os.chmod", lambda path, mode: calls.append((path, mode)))

    def remove(path):
        calls.append(("remove", path))

    _clear_readonly_and_retry(remove, "locked-file", (None, PermissionError("denied"), None))
    assert calls[0][0] == "locked-file"
    assert calls[1] == ("remove", "locked-file")


def test_clear_readonly_checkout_entry_propagates_non_permission_errors():
    from eval.gpu_batch import _clear_readonly_and_retry

    with pytest.raises(OSError, match="disk"):
        _clear_readonly_and_retry(lambda path: None, "bad-file", (None, OSError("disk"), None))


def test_mock_result_has_wrapped_eval_shape():
    tmp, path = _queue_file([
        {"pr": 9, "title": "mock me", "author": "alice", "head_sha": "f" * 40,
         "position": 1},
    ])
    try:
        payload = mock_result(load_queue(path)[0], EvalSpec(transforms="mine"))
        assert payload["mock"] is True
        assert payload["eval"]["best"] == "mine"
        assert payload["eval"]["config"]["device"] == "RTX 5070 Ti (mock)"
        assert payload["eval"]["transforms"]["mine"]["improvement"] is True
    finally:
        tmp.cleanup()


def test_wrap_result_adds_pr_metadata():
    tmp, path = _queue_file([
        {"pr": 5, "title": "real", "author": "bob", "head_sha": "a" * 40,
         "position": 1},
    ])
    try:
        item = load_queue(path)[0]
        payload = wrap_result(item, '{"config": {}, "transforms": {}, "best": null}')
        assert payload["pr"] == 5
        assert payload["mock"] is False
        assert "eval" in payload
    finally:
        tmp.cleanup()


def test_run_item_mock_writes_result_without_checkout():
    tmp, path = _queue_file([
        {"pr": 8, "title": "mock run", "author": "carol", "head_sha": "b" * 40,
         "position": 1},
    ])
    try:
        with tempfile.TemporaryDirectory() as d:
            out = run_item(
                load_queue(path)[0],
                repo="owner/repo",
                workdir=Path(d) / "work",
                results_dir=Path(d) / "results",
                spec=EvalSpec(transforms="mine"),
                mock=True,
            )
            data = json.loads(out.read_text())
            assert data["pr"] == 8
            assert data["eval"]["best"] == "mine"
            assert not (Path(d) / "work").exists()
    finally:
        tmp.cleanup()


def test_run_item_skips_undeclared_feature_without_checkout(tmp_path):
    item = QueueItem(pr=10, title="undeclared", author="alice", head_sha="a" * 40)
    out = run_item(
        item,
        repo="owner/repo",
        workdir=tmp_path / "work",
        results_dir=tmp_path / "results",
        spec=EvalSpec(),
    )
    data = json.loads(out.read_text())
    assert data["state"] == "missing_evaluation_declaration"
    assert not (tmp_path / "work").exists()


def test_load_queue_reads_track():
    tmp, path = _queue_file([
        {"pr": 5, "head_sha": "s", "track": "low-rank"},
        {"pr": 6, "head_sha": "t"},   # no track -> None
    ])
    try:
        items = {it.pr: it for it in load_queue(path)}
        assert items[5].track == "low-rank"
        assert items[6].track is None
    finally:
        tmp.cleanup()


def test_spec_for_track_pins_the_regime():
    from eval.gpu_batch import spec_for_track
    base = EvalSpec(n=8192, rank_m=999, fill="random", data_rank=None)
    lr = spec_for_track(base, "low-rank")
    # the PR's chosen rank_m/fill are IGNORED -> pinned low-rank regime
    assert (lr.fill, lr.data_rank, lr.rank_m) == ("lowrank", 16, 64)
    # unknown / unspecified track -> unchanged (full-rank reference fallback)
    assert spec_for_track(base, None) is base
    assert spec_for_track(base, "no-such-track") is base


def _run_cell(acc, lat, dom):
    return {"accuracy": acc, "latency_s": lat, "peak_vram_bytes": 100,
            "peak_vram_mib": 1.0, "flop_ratio_vs_exact": 10.0,
            "faster_than_exact": dom, "less_vram_than_exact": True,
            "fewer_flops_than_exact": True, "gated": not dom,
            "improvement": dom, "score": 5.0 if dom else 0.0}


def _run_out(seed, cell):
    return {"config": {"seed": seed}, "exact": {}, "transforms": {"ny": cell}}


def test_aggregate_admits_only_if_every_run_dominates():
    from eval.gpu_batch import aggregate_runs
    # all 3 seeds dominate -> admitted, worst-case metrics, min score
    agg = aggregate_runs([
        _run_out(1, _run_cell(0.99, 0.20, True)),
        _run_out(2, _run_cell(0.97, 0.25, True)),
        _run_out(3, _run_cell(0.98, 0.22, True)),
    ])
    ny = agg["transforms"]["ny"]
    assert ny["improvement"] is True
    assert ny["accuracy"] == 0.97 and ny["latency_s"] == 0.25   # min acc, max latency
    assert ny["score"] == 5.0 and ny["runs"] == 3
    assert agg["aggregation"]["runs"] == 3


def test_aggregate_rejects_lucky_seed():
    from eval.gpu_batch import aggregate_runs
    # one seed fails the gate -> NOT admitted, score forced to 0
    agg = aggregate_runs([
        _run_out(1, _run_cell(0.99, 0.20, True)),
        _run_out(2, _run_cell(0.90, 0.60, False)),   # loses on this seed
        _run_out(3, _run_cell(0.98, 0.22, True)),
    ])
    assert agg["transforms"]["ny"]["improvement"] is False
    assert agg["transforms"]["ny"]["score"] == 0.0


_TRANSFORMS_SRC = '''\
class Transform:
    name = "base"

class RandomizedSVDTransform(Transform):
    name = "rsvd"
    def basis(self, n, m):
        return qr(n, m)

class NystromTransform(Transform):
    name = "nystrom"
    def basis(self, n, m):
        return cols(n, m)
'''

_UPDATE_RSVD_DIFF = '''\
--- a/strategy/transforms.py
+++ b/strategy/transforms.py
@@ -6,2 +6,2 @@ class RandomizedSVDTransform(Transform):
-    def basis(self, n, m):
-        return qr4(n, m)
+    def basis(self, n, m):
+        return qr(n, m)
'''

_ADD_NYSTROM_DIFF = '''\
--- a/strategy/transforms.py
+++ b/strategy/transforms.py
@@ -8,0 +9,4 @@
+class NystromTransform(Transform):
+    name = "nystrom"
+    def basis(self, n, m):
+        return cols(n, m)
'''

_UNRELATED_DIFF = '''\
--- a/tests/test_x.py
+++ b/tests/test_x.py
@@ -1 +1 @@
-x
+y
'''


def test_transform_touched_new_and_update_validate_equally():
    from eval.gpu_batch import transform_touched_in
    # UPDATE: modifying rsvd's class body counts, even though the string "rsvd"
    # never appears in the changed lines -- this is the #156 case that the old
    # name-in-diff check wrongly rejected.
    assert transform_touched_in(_TRANSFORMS_SRC, _UPDATE_RSVD_DIFF, "rsvd") is True
    assert transform_touched_in(_TRANSFORMS_SRC, _UPDATE_RSVD_DIFF, "nystrom") is False
    # NEW: adding the nystrom class counts -- the #194 case.
    assert transform_touched_in(_TRANSFORMS_SRC, _ADD_NYSTROM_DIFF, "nystrom") is True
    # claiming a transform the PR does not touch -> not verified.
    assert transform_touched_in(_TRANSFORMS_SRC, _UNRELATED_DIFF, "rsvd") is False


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
