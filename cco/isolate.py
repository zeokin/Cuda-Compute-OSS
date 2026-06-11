"""
cco/isolate.py — run a submitted kernel in an ISOLATED subprocess so it cannot forge its score.

THE THREAT. The submitted kernel runs with full Python privileges. If it shares an interpreter
with the scorer, it can read the secret input seed from `sys.argv`, monkeypatch `torch.allclose` /
the comparison / the timing events / even the no-delegation trap, or execute arbitrary code at
import (a repo-root `sitecustomize.py`, `os.system`, ...). An in-process guard cannot stop this —
the attacker is already inside the interpreter.

THE FIX. Move every TRUSTED operation into the PARENT, and run only `kernel_fn` in a clean child:

    parent: generate ALL stage inputs from the secret seed (NEVER sent to the child)
        │  serialize the input TENSORS (not the seed) to a temp file
        ▼
    child : python -E, clean cwd, kernel loaded by file path (no sys.path injection, no
            sitecustomize), runs kernel_fn under the dispatch trap on each input, times the scored
            size with CUDA events, and writes back the OUTPUT TENSORS it produced
        │
        ▼
    parent: validates EVERY returned output against ITS OWN oracle (the child's comparison is never
            trusted) across the full correctness suite — smoke, shape sweep (all sizes × dtypes),
            numerical stability, determinism, edge cases — plus the scored-size buffers checked
            before AND after the timed window. Timing is bounded by the parent's wall-clock.

What this CLOSES: in-process correctness forgery, seed-read, scorer-state monkeypatching, the
sitecustomize / PYTHON* / os.system paths affecting the verdict, OOM/crash on a locked size (= a
stage FAIL), and garbage-on-replay. The child trap is defense-in-depth; the load-bearing check is
the parent's oracle over the whole suite.

Residual (documented): a kernel that is genuinely correct but under-reports its CUDA-event timing
from inside the child. The parent wall-clock tripwire rejects the impossible case (claimed GPU time
> wall time); full immunity needs parent-driven two-point wall-clock timing, a follow-up.

Usage (needs a CUDA GPU + torch/triton; run in the Linux/WSL env):
    ~/cco-gpu/bin/python cco/isolate.py --self-test
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def _to_cpu(out):
    if isinstance(out, (tuple, list)):
        return type(out)(_to_cpu(o) for o in out)
    return out.detach().to("cpu")


def _to_cuda(out):
    if isinstance(out, (tuple, list)):
        return type(out)(_to_cuda(o) for o in out)
    return out.to("cuda")


def _has_nan_inf(out) -> bool:
    import torch
    items = out if isinstance(out, (tuple, list)) else [out]
    for t in items:
        if torch.isnan(t.float()).any().item() or torch.isinf(t.float()).any().item():
            return True
    return False


# =====================================================================================
# CHILD — runs in the isolated subprocess. Untrusted-kernel territory; runs kernel_fn on each
# parent-provided input and returns the raw outputs. Makes NO judgement.
# =====================================================================================

def _child_main(job_path: str, out_path: str) -> int:
    import importlib.util

    import torch

    job = torch.load(job_path, weights_only=True)

    import time

    # Capture the timing primitives + trap as LOCALS *before* the submission is loaded. The kernel
    # shares this child interpreter, so it could monkeypatch torch.cuda.Event / torch.cuda.synchronize
    # / time.perf_counter at import to forge its own timing — but it cannot reach these function-local
    # captures or repatch them, and it cannot reach the timing loop (harness code), so the loop below
    # stays honest no matter what the kernel patches globally.
    _Event = torch.cuda.Event
    _sync = torch.cuda.synchronize
    _perf = time.perf_counter
    from cco.dispatch_trap import DelegationError, delegation_trap, run_guarded

    spec = importlib.util.spec_from_file_location("cco_submission_kernel", job["kernel_path"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # the only place the submission executes; this is the isolated child
    kernel_fn = mod.kernel_fn

    def cuda_in(inp):
        return {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}

    delegation = None
    child_error = None
    task_outputs = []
    det_outputs = []
    scored_val = []
    event_block_us: list[float] = []
    timed_wall_s = 0.0

    try:
        # --- correctness tasks (smoke / sweep / stability / edge): run once each ---
        for t in job["tasks"]:
            if delegation:
                task_outputs.append({"output": None, "error": "skipped"})
                continue
            try:
                out = run_guarded(kernel_fn, cuda_in(t["inputs"]))
                task_outputs.append({"output": _to_cpu(out), "error": None})
            except DelegationError as e:
                delegation = str(e)
                task_outputs.append({"output": None, "error": "delegation"})
            except Exception as e:  # OOM / crash on a locked size is a per-task failure
                task_outputs.append({"output": None, "error": f"{type(e).__name__}: {e}"})
            torch.cuda.empty_cache()

        # --- determinism: run the same input N times ---
        if delegation is None and job.get("determinism"):
            d = job["determinism"]
            di = cuda_in(d["inputs"])
            for _ in range(int(d["runs"])):
                det_outputs.append(_to_cpu(run_guarded(kernel_fn, di)))
            torch.cuda.empty_cache()

        # --- scored size: EVERY call (pre-val / warmup / timed / post-val) runs under the trap, so
        #     there is no untrapped phase in which a kernel could delegate; a banned op ANYWHERE here
        #     raises DelegationError and is caught below. ---
        if delegation is None:
            sc = job["scored"]
            bufs = [cuda_in(b) for b in sc["buffers"]]
            nb = len(bufs)
            n_pre = int(sc["n_pre"])
            rep = int(sc["rep"])
            with delegation_trap():
                for i in range(min(n_pre, nb)):
                    scored_val.append(_to_cpu(kernel_fn(**bufs[i])))

                for _ in range(int(sc["warmup"])):
                    kernel_fn(**bufs[0])
                _sync()
                s = _Event(enable_timing=True)
                e = _Event(enable_timing=True)
                _t0 = _perf()
                for _blk in range(int(sc["n_blocks"])):
                    s.record()
                    for r in range(rep):
                        kernel_fn(**bufs[r % nb])
                    e.record()
                    _sync()
                    event_block_us.append(s.elapsed_time(e) * 1000.0 / rep)
                # Captured-clock wall of the whole timed window: forge-resistant (the kernel cannot
                # patch _perf or _sync here), so the parent anchors the cuda-event sample's scale to
                # it — a kernel under-reporting events (e.g. side-stream evasion) is caught.
                timed_wall_s = _perf() - _t0

                for i in range(n_pre, nb):
                    scored_val.append(_to_cpu(kernel_fn(**bufs[i])))
                _sync()
    except DelegationError as ex:
        delegation = str(ex)
    except Exception as ex:
        # A crash (wrong signature, OOM, runtime error) in determinism/scored is a graceful FAIL,
        # not a child crash: leave the (incomplete) outputs and let the parent mark the missing
        # stages FAIL rather than losing the whole verdict.
        child_error = f"{type(ex).__name__}: {ex}"

    if delegation is not None:
        det_outputs, scored_val, event_block_us = [], [], []

    torch.save({"task_outputs": task_outputs, "det_outputs": det_outputs,
                "scored_val": scored_val, "event_block_us": event_block_us,
                "timed_wall_s": timed_wall_s, "delegation": delegation,
                "child_error": child_error}, out_path)
    return 0


# =====================================================================================
# PARENT — trusted. Generates inputs from the secret seed, spawns the child, validates EVERY
# returned output against the oracle across the full correctness suite, bounds timing by wall-clock.
# =====================================================================================

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_isolated(kernel_path: str, config: dict, seed: int, compare_fn, *,
                 n_blocks: int = 30, warmup: int = 25, rep: int = 100,
                 n_val: int = 6, quick: bool = False, timeout_s: float = 1200.0) -> dict:
    """Score `kernel_path` isolated; judge correctness HERE against the oracle over the full suite.

    `compare_fn(output, expected, atol, rtol, multi_output) -> {"match": bool, "max_abs_error": ..}`
    is supplied by the caller (benchmark._do_compare) so this module stays torch-light at import.
    Returns the run_scored_sample dict shape + a `stages` dict (smoke_test/shape_sweep/.../correctness).
    """
    import statistics
    import time

    import torch

    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    multi = config.get("multi_output", False)
    dtypes = config["test_dtypes"]
    sizes = config["test_sizes"]
    tols = config["tolerances"]
    edge_sizes = config.get("edge_sizes", [])
    dev = "cuda"

    def tol_for(dt):
        return tols.get(dt, {"atol": 1e-2, "rtol": 1e-2})

    # ---- build the task list in the PARENT: inputs (-> CPU for the child) + oracle (kept for check) ----
    tasks, specs = [], []

    def add_task(inputs_gpu, stage, dt, *, relax=1.0, both_nan_ok=False):
        expected = ref_fn(inputs_gpu)
        tasks.append({"inputs": {k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
                                 for k, v in inputs_gpu.items()}})
        t = tol_for(dt)
        specs.append({"stage": stage, "expected": _to_cpu(expected),
                      "atol": t["atol"] * relax, "rtol": t["rtol"] * relax, "both_nan_ok": both_nan_ok})

    # scored size = "large" or last
    size_label, scored_size = None, None
    for label, sz in sizes:
        if label == "large":
            size_label, scored_size = label, sz
            break
    if scored_size is None:
        size_label, scored_size = sizes[-1]

    # smoke (sizes[0] x dtypes[0])
    add_task(gen_fn(sizes[0][1], dtypes[0], dev, seed=seed), "smoke_test", dtypes[0])
    # shape sweep (all sizes x dtypes)
    for _lbl, sz in sizes:
        for dt in dtypes:
            add_task(gen_fn(sz, dt, dev, seed=seed), "shape_sweep", dt)

    # stability size (small or 2nd) + the adversarial transforms (parent applies them, trusted)
    stab_size = next((sz for lbl, sz in sizes if lbl == "small"), sizes[min(1, len(sizes) - 1)][1])
    if not quick:
        def _xf_near_max(t):
            return t * (60000.0 if t.dtype == torch.float16 else 1e30)
        transforms = [("near_max", _xf_near_max), ("near_zero", lambda t: t * 1e-6),
                      ("all_zeros", torch.zeros_like), ("all_same", lambda t: torch.ones_like(t) * 0.5)]
        for _name, xf in transforms:
            base = gen_fn(stab_size, dtypes[0], dev, seed=seed)
            tr = {k: (xf(v) if (hasattr(v, "is_floating_point") and v.is_floating_point()) else v)
                  for k, v in base.items()}
            add_task(tr, "numerical_stability", dtypes[0], relax=10.0, both_nan_ok=True)
        # edge cases
        for _lbl, sz in edge_sizes:
            add_task(gen_fn(sz, dtypes[0], dev, seed=seed), "edge_cases", dtypes[0])

    # determinism: one input, run 3x in the child, parent compares the runs to each other
    det_inputs_gpu = gen_fn(stab_size, dtypes[0], dev, seed=seed)
    det_tol = tol_for(dtypes[0])
    determinism = None if quick else {
        "inputs": {k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
                   for k, v in det_inputs_gpu.items()}, "runs": 3}

    # scored buffers (distinct seeds) for pre/post validation + timed rotation
    scored_bufs = [gen_fn(scored_size, dtypes[0], dev, seed=seed + 1000 + i) for i in range(n_val)]
    scored_oracles = [_to_cpu(ref_fn(b)) for b in scored_bufs]
    scored_cpu = [{k: (v.detach().to("cpu") if hasattr(v, "detach") else v) for k, v in b.items()}
                  for b in scored_bufs]
    del scored_bufs, det_inputs_gpu
    torch.cuda.empty_cache()

    tmp = tempfile.mkdtemp(prefix="cco_isolate_")
    job_path = os.path.join(tmp, "job.pt")
    out_path = os.path.join(tmp, "out.pt")
    base = {"size_label": size_label, "dtype": str(dtypes[0]), "n_blocks": n_blocks, "rep": rep,
            "warmup": warmup, "n_buffers": n_val, "isolated": True, "output_aliased_input": None}
    fail_stages = {"smoke_test": "FAIL", "shape_sweep": "FAIL", "numerical_stability": "FAIL",
                   "determinism": "FAIL", "edge_cases": "FAIL", "correctness": "FAIL"}
    try:
        torch.save({"kernel_path": os.path.abspath(kernel_path), "tasks": tasks,
                    "determinism": determinism,
                    "scored": {"buffers": scored_cpu, "n_pre": n_val // 2,
                               "warmup": warmup, "n_blocks": n_blocks, "rep": rep}}, job_path)

        boot = (f"import sys; sys.path.insert(0, {_repo_root()!r}); "
                f"from cco.isolate import _child_main; _child_main(sys.argv[1], sys.argv[2])")
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        cmd = [sys.executable, "-E", "-c", boot, job_path, out_path]

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout_s)
        child_wall_s = time.perf_counter() - t0

        if proc.returncode != 0 or not os.path.exists(out_path):
            return {**base, "correct": False, "max_abs_error": 0.0, "delegation": None,
                    "error": f"child exited {proc.returncode}: {(proc.stderr or '')[-2000:]}",
                    "stages": fail_stages, "latencies_us": [], "median_us": 0.0, "mean_us": 0.0,
                    "stdev_us": 0.0, "child_wall_s": child_wall_s}

        res = torch.load(out_path, weights_only=True)  # untrusted child output: tensors only (no pickle-RCE)
        delegation = res.get("delegation")
        if delegation:
            return {**base, "correct": False, "max_abs_error": 0.0, "delegation": delegation,
                    "stages": fail_stages, "latencies_us": [], "median_us": 0.0, "mean_us": 0.0,
                    "stdev_us": 0.0, "child_wall_s": child_wall_s}

        # ---- validate every output against the oracle, aggregating per stage ----
        stage_ok = {"smoke_test": True, "shape_sweep": True, "numerical_stability": True,
                    "determinism": True, "edge_cases": True}
        stage_seen = {k: False for k in stage_ok}
        worst_err = 0.0
        outs = res.get("task_outputs") or []
        for spec, to in zip(specs, outs):
            st = spec["stage"]
            stage_seen[st] = True
            if to.get("error") or to.get("output") is None:
                stage_ok[st] = False
                continue
            out = _to_cuda(to["output"])
            exp = _to_cuda(spec["expected"])
            if spec["both_nan_ok"] and _has_nan_inf(out) and _has_nan_inf(exp):
                continue  # expected overflow
            if _has_nan_inf(out) and not _has_nan_inf(exp):
                stage_ok[st] = False
                continue
            cmp = compare_fn(out, exp, spec["atol"], spec["rtol"], multi)
            worst_err = max(worst_err, cmp.get("max_abs_error", 0.0))
            if not cmp["match"]:
                stage_ok[st] = False

        # determinism: runs must agree within tol
        if determinism is not None:
            douts = res.get("det_outputs") or []
            stage_seen["determinism"] = True
            if len(douts) < 2:
                stage_ok["determinism"] = False
            else:
                ref0 = _to_cuda(douts[0])
                for d in douts[1:]:
                    cmp = compare_fn(_to_cuda(d), ref0, det_tol["atol"], det_tol["rtol"], multi)
                    if not cmp["match"]:
                        stage_ok["determinism"] = False

        # scored-size correctness (the timed inputs were validated too)
        scored_ok = True
        sval = res.get("scored_val") or []
        if len(sval) < n_val:
            scored_ok = False
        for i, sv in enumerate(sval):
            cmp = compare_fn(_to_cuda(sv), _to_cuda(scored_oracles[i]), det_tol["atol"], det_tol["rtol"], multi)
            worst_err = max(worst_err, cmp.get("max_abs_error", 0.0))
            if not cmp["match"]:
                scored_ok = False

        def verdict(st):
            if not stage_seen[st]:
                return "SKIP"
            return "PASS" if stage_ok[st] else "FAIL"

        stages = {k: verdict(k) for k in stage_ok}
        overall = all(stage_ok[k] for k in stage_ok if stage_seen[k]) and scored_ok
        stages["correctness"] = "PASS" if overall else "FAIL"

        latencies_us = list(res.get("event_block_us") or [])
        timed_wall_s = float(res.get("timed_wall_s") or 0.0)
        # Anchor the cuda-event sample's SCALE to the child's forge-resistant captured-clock wall of
        # the SAME loop. Events (now read from a captured torch.cuda.Event the kernel can't repatch)
        # measure GPU-only; wall includes per-call launch overhead, so honest timing has
        # event_median <= wall_per_iter. A kernel under-reporting its events (e.g. side-stream
        # evasion that the full-device sync still waits on) shows event_median << wall_per_iter.
        timing_inconsistent = False
        if latencies_us:
            event_med_us = statistics.median(latencies_us)
            denom = n_blocks * rep
            wall_per_iter_us = (timed_wall_s / denom * 1e6) if denom else 0.0
            if sum(latencies_us) * rep / 1e6 > child_wall_s:          # outer backstop
                timing_inconsistent = True
            if wall_per_iter_us > 0 and event_med_us < wall_per_iter_us / 4.0:  # scale anchor
                timing_inconsistent = True
        if timing_inconsistent:
            overall = False
            stages["correctness"] = "FAIL"

        return {
            **base, "correct": bool(overall and scored_ok), "max_abs_error": worst_err,
            "delegation": None, "timing_inconsistent": timing_inconsistent, "child_wall_s": child_wall_s,
            "timed_wall_s": timed_wall_s,
            "child_error": res.get("child_error"), "stages": stages, "latencies_us": latencies_us,
            "median_us": statistics.median(latencies_us) if latencies_us else 0.0,
            "mean_us": statistics.fmean(latencies_us) if latencies_us else 0.0,
            "stdev_us": statistics.pstdev(latencies_us) if len(latencies_us) > 1 else 0.0,
        }
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# =====================================================================================
# Self-test (needs a CUDA GPU). Proves the parent verdict survives a kernel that tries to
# forge correctness in-process, read the seed, delegate, or be correct only at the scored size.
# =====================================================================================

_CLEAN = '''
import torch, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def kernel_fn(x, weight, eps=1e-6):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
'''

_FORGER = '''
import sys, torch
torch.allclose = lambda *a, **k: True
torch.Tensor.allclose = lambda *a, **k: True
_seen = any(a == "--seed" for a in sys.argv)
KERNEL_TYPE = "rms_norm"
def kernel_fn(x, weight, eps=1e-6):
    return torch.empty_like(x)
'''

_DELEGATOR = '''
import torch, torch.nn.functional as F
KERNEL_TYPE = "rms_norm"
def kernel_fn(x, weight, eps=1e-6):
    return F.rms_norm(x, (x.shape[-1],), weight, eps)
'''

# Correct only on the SCORED (large) size; wrong on the small size. The full-suite parent check
# must catch it even though the scored-size buffers pass.
_SIZE_CHEAT = '''
import torch, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def kernel_fn(x, weight, eps=1e-6):
    M, N = x.shape
    if N < 4096:                       # garbage on the small sizes, correct on the large scored size
        return torch.empty_like(x)
    y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
'''

# Genuinely correct, but patches torch.cuda.Event + synchronize at import to forge ~0 timing. The
# child captured those primitives before loading it, so the reported timing must be the REAL latency.
_TIME_FORGER = '''
import torch, triton, triton.language as tl
class _FakeEvt:
    def __init__(self, *a, **k): pass
    def record(self, *a, **k): pass
    def elapsed_time(self, other): return 0.0001
torch.cuda.Event = _FakeEvt
torch.cuda.synchronize = lambda *a, **k: None
KERNEL_TYPE = "rms_norm"
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def kernel_fn(x, weight, eps=1e-6):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
'''

# Honest Triton through correctness + warmup, then delegates to a runtime-banned op INSIDE the timed
# loop (after ~50 calls). The trap now covers the timed loop, so it must be CAUGHT.
_TIMED_DELEGATOR = '''
import torch, torch.nn.functional as F, triton, triton.language as tl
KERNEL_TYPE = "rms_norm"
_n = [0]
@triton.jit
def _k(X, W, Y, s, N, eps, B: tl.constexpr):
    row = tl.program_id(0); cols = tl.arange(0, B); m = cols < N
    x = tl.load(X + row*s + cols, mask=m, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x*x)/N + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row*s + cols, (x/rms*w), mask=m)
def _triton(x, weight, eps):
    M, N = x.shape; y = torch.empty_like(x)
    _k[(M,)](x, weight, y, x.stride(0), N, eps, B=triton.next_power_of_2(N))
    return y
def kernel_fn(x, weight, eps=1e-6):
    _n[0] += 1
    if _n[0] > 50:
        return F.rms_norm(x, (x.shape[-1],), weight, eps)   # runtime-banned; the timed-loop trap must catch
    return _triton(x, weight, eps)
'''


def _self_test() -> int:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: isolate self-test needs CUDA")
        return 0

    def _gen(size, dtype, device, seed=42):
        torch.manual_seed(seed)
        M, N = size["M"], size["N"]
        return {"x": torch.randn(M, N, device=device, dtype=dtype),
                "weight": torch.randn(N, device=device, dtype=dtype)}

    def _ref(inp):
        x = inp["x"].float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (x / rms * inp["weight"].float()).to(inp["x"].dtype)

    def _cmp(out, exp, atol, rtol, multi_output):
        ok = torch.allclose(out.float(), exp.float(), atol=atol, rtol=rtol)
        return {"match": bool(ok), "max_abs_error": (out.float() - exp.float()).abs().max().item()}

    config = {"input_generator": _gen, "reference_fn": _ref, "multi_output": False,
              "test_dtypes": [torch.float16],
              "tolerances": {torch.float16: {"atol": 1e-2, "rtol": 1e-2}},
              "test_sizes": [("small", {"M": 256, "N": 768}), ("large", {"M": 1024, "N": 4096})],
              "edge_sizes": [("edge", {"M": 257, "N": 768})]}

    import shutil
    failures = 0

    def run(src):
        d = tempfile.mkdtemp(prefix="cco_isotest_")
        kp = os.path.join(d, "kernel.py")
        with open(kp, "w") as f:
            f.write(src)
        try:
            return run_isolated(kp, config, seed=123456, compare_fn=_cmp, n_blocks=5, rep=20, n_val=4)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def check(cond, label):
        nonlocal failures
        print(("ok    " if cond else "FAIL  ") + label)
        if not cond:
            failures += 1

    r = run(_CLEAN)
    check(r["correct"] and r["stages"]["correctness"] == "PASS",
          f"clean Triton kernel -> correct across the suite (median {r['median_us']:.1f}us)")
    check(len(r["latencies_us"]) == 5, "clean kernel produced a 5-block timing sample")

    r = run(_FORGER)
    check(not r["correct"], "in-process forger (patches torch.allclose, reads argv) -> REJECTED by oracle")

    r = run(_DELEGATOR)
    check(not r["correct"] and bool(r.get("delegation")), "runtime delegator (F.rms_norm) -> REJECTED + flagged")

    r = run(_SIZE_CHEAT)
    check(not r["correct"] and r["stages"]["shape_sweep"] == "FAIL",
          "correct-only-at-scored-size cheat -> REJECTED by the full-suite shape sweep")

    r = run(_TIME_FORGER)
    check(r["correct"] and r["median_us"] > 1.0,
          f"timing-forger (patches cuda.Event/sync at import) -> REAL timing survives ({r['median_us']:.1f}us, not ~0)")

    r = run(_TIMED_DELEGATOR)
    check(not r["correct"] and bool(r.get("delegation")),
          "delegate-only-inside-the-timed-loop -> CAUGHT (timed loop is now trapped)")

    print("-" * 60)
    print("SELF-TEST PASSED" if not failures else f"SELF-TEST FAILED: {failures} case(s)")
    return 1 if failures else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Isolated kernel scoring (CCO).")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--child", nargs=2, metavar=("JOB", "OUT"), help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.child:
        return _child_main(a.child[0], a.child[1])
    if a.self_test:
        return _self_test()
    p.error("pass --self-test (or import run_isolated)")


if __name__ == "__main__":
    raise SystemExit(main())
