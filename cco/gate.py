"""
cco/gate.py — Gate 3 cheap static checks (Step 15).

The gate pipeline runs these on a submitted kernel.py BEFORE any GPU spend (default verdict =
reject). Three deterministic checks bundled into one verdict:

  1. size cap     — kernel.py <= `kernel_py_size_cap_bytes` (no embedded lookup tables / data blobs);
  2. declared track — KERNEL_TYPE is read STATICALLY (no execution of untrusted code), must be a
                    known track AND match the PR payload's `kernel_type`. This stops a miner from
                    declaring an easier track, or being scored against the wrong oracle/champion;
  3. no delegation — the static AST scan (cco/guard_kernel.py).

On a win, the **CCO maintainer bot** (a maintainer-owned token — an owner/collaborator account, or a
GitHub App honored via `trusted_label_pipeline`, NOT the read-only Gittensor App) merges the PR and
moves the `cco-winner-<kernel_type>` label onto it; SN74 validators only observe the merged + labeled
result. That orchestration is the bot/pipeline's, not this module's.

Usage:
    uv run --no-project python cco/gate.py --self-test
    uv run --no-project python cco/gate.py kernel.py <payload_kernel_type> [--config cco.config.json]
"""

from __future__ import annotations

import json
import os
import sys

# Make `import cco.*` work whether this file is run as a script or imported as a package module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cco.guard_kernel import (  # noqa: E402
    DEFAULT_POLICY,
    extract_kernel_type,
    load_policy_from_config,
    scan_file,
)

DEFAULT_TRACKS = ["rms_norm", "matmul", "qkv_part_rope", "swiglu_input_quant", "dsa_forward"]
DEFAULT_SIZE_CAP = 65536


def gate3(kernel_path: str, payload_kernel_type: str, config_path: "str | None" = None) -> dict:
    """Run the three cheap Gate-3 checks; return a structured verdict (default = reject)."""
    tracks, size_cap, policy = DEFAULT_TRACKS, DEFAULT_SIZE_CAP, DEFAULT_POLICY
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tracks = cfg.get("tracks", tracks)
        size_cap = cfg.get("kernel_py_size_cap_bytes", size_cap)
        if "no_delegation" in cfg:  # else keep DEFAULT_POLICY (== the canonical denylist)
            policy = load_policy_from_config(config_path)

    reasons: list[str] = []

    with open(kernel_path, "rb") as f:
        raw = f.read()
    size_ok = len(raw) <= size_cap
    if not size_ok:
        reasons.append(f"kernel.py is {len(raw)} bytes, exceeds cap {size_cap}")

    declared = extract_kernel_type(raw.decode("utf-8", errors="replace"))
    if declared is None:
        track_ok = False
        reasons.append("no static KERNEL_TYPE string literal in kernel.py")
    elif declared not in tracks:
        track_ok = False
        reasons.append(f"KERNEL_TYPE {declared!r} is not a known track")
    elif declared != payload_kernel_type:
        track_ok = False
        reasons.append(f"KERNEL_TYPE {declared!r} != payload kernel_type {payload_kernel_type!r}")
    else:
        track_ok = True

    vios = scan_file(kernel_path, policy)
    delegation_ok = len(vios) == 0
    if not delegation_ok:
        reasons.append(f"{len(vios)} no-delegation violation(s)")

    return {
        "pass": bool(size_ok and track_ok and delegation_ok),
        "kernel_type": declared,
        "checks": {"size_ok": size_ok, "track_ok": track_ok, "delegation_ok": delegation_ok},
        "violations": [v.as_dict() for v in vios],
        "reasons": reasons,
    }


# --------------------------------------------------------------------------------------
# Self-test (pure Python; temp kernels)
# --------------------------------------------------------------------------------------

_CLEAN = '''
import torch
import triton
import triton.language as tl
KERNEL_TYPE = "rms_norm"

@triton.jit
def _k(X, Y, N, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    tl.store(Y + cols, x, mask=cols < N)

def kernel_fn(x, weight, eps=1e-6):
    y = torch.empty_like(x)
    _k[(x.shape[0],)](x, y, x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]))
    return y
'''

_DELEGATING = 'import torch\nKERNEL_TYPE = "matmul"\ndef kernel_fn(a, b):\n    return torch.matmul(a, b)\n'
_WRONG_TRACK = _CLEAN.replace('KERNEL_TYPE = "rms_norm"', 'KERNEL_TYPE = "not_a_track"')
_NO_TYPE = _CLEAN.replace('KERNEL_TYPE = "rms_norm"\n', '')


def _self_test() -> int:
    import tempfile
    failures = 0

    def check(cond, label):
        nonlocal failures
        print(("ok   " if cond else "FAIL ") + label)
        if not cond:
            failures += 1

    tmp = tempfile.mkdtemp(prefix="cco_gate_")
    cfg_path = os.path.join(tmp, "cco.config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"tracks": DEFAULT_TRACKS, "kernel_py_size_cap_bytes": 200}, f)  # tiny cap to test size

    def write(name, content):
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return p

    clean = write("clean.py", _CLEAN)
    # 1. clean kernel, payload matches declared -> PASS (no config -> default 64KB cap)
    r = gate3(clean, "rms_norm")
    check(r["pass"] and r["kernel_type"] == "rms_norm", "clean + matching payload -> PASS")
    # 2. clean kernel, payload declares a DIFFERENT track -> reject (track mismatch)
    r = gate3(clean, "matmul")
    check(not r["pass"] and not r["checks"]["track_ok"], "declared/payload track mismatch -> reject")
    # 3. delegating kernel -> reject (delegation)
    r = gate3(write("cheat.py", _DELEGATING), "matmul")
    check(not r["pass"] and not r["checks"]["delegation_ok"], "delegation -> reject")
    # 4. unknown track -> reject
    r = gate3(write("wrong.py", _WRONG_TRACK), "not_a_track")
    check(not r["pass"] and not r["checks"]["track_ok"], "unknown track -> reject")
    # 5. no KERNEL_TYPE -> reject
    r = gate3(write("notype.py", _NO_TYPE), "rms_norm")
    check(not r["pass"] and r["kernel_type"] is None, "missing KERNEL_TYPE -> reject")
    # 6. size cap (tiny cap from cfg) -> reject on size
    r = gate3(clean, "rms_norm", config_path=cfg_path)
    check(not r["pass"] and not r["checks"]["size_ok"], "over size cap -> reject")

    print("-" * 60)
    print("SELF-TEST PASSED" if not failures else f"SELF-TEST FAILED: {failures} case(s)")
    return 1 if failures else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Gate-3 cheap static checks for CCO submissions.")
    p.add_argument("kernel", nargs="?", help="path to kernel.py")
    p.add_argument("kernel_type", nargs="?", help="the payload's declared kernel_type")
    p.add_argument("--config", help="cco.config.json (tracks, size cap, denylist)")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return _self_test()
    if not a.kernel or not a.kernel_type:
        p.error("provide <kernel.py> <payload_kernel_type>, or --self-test")
    verdict = gate3(a.kernel, a.kernel_type, a.config)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
