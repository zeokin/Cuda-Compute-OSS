"""
cco/dispatch_trap.py — Runtime no-delegation trap for CCO competition submissions (Step 2).

The static AST guard (cco/guard_kernel.py) is necessary but not sufficient: a determined
miner can construct a delegating call the scanner can't see (obfuscated dispatch, a call
built at runtime, an op reached through an alias). This module is the runtime backstop.

It runs the miner's `kernel_fn(**inputs)` under TWO nested interception layers:

  * a `TorchFunctionMode` — catches HIGH-LEVEL public ops by name at the call site, BEFORE
    any decomposition: `torch.matmul`, `F.silu`, `F.scaled_dot_product_attention`, the `@`
    operator (`Tensor.__matmul__`), `x.softmax(...)`, etc. This catches composite ops (like
    SDPA) that decompose into a fused backend op below the dispatcher.
  * a `TorchDispatchMode` — catches ATen ops by name (`aten::mm`, `aten::_softmax`,
    `aten::_scaled_dot_product_flash_attention`, ...) reached indirectly, e.g. through
    `torch.ops.aten.*` which bypasses the function layer.

If either layer sees a banned "this op IS the computation" op, the call is rejected with a
`DelegationError`. Crucially, **a real `@triton.jit` launch goes through NEITHER layer** — it
is a custom CUDA kernel, invisible here — so a legitimate Triton kernel runs clean (only
allocation/view ops like `empty_like`/`reshape` appear, which are allowed), while
`return torch.matmul(a, b)` or `F.silu(x)` is caught even if it slipped past the static scan.
cuBLAS/cuDNN reached via torch are caught (they surface as `aten::mm`/`aten::convolution`);
an LD_PRELOAD symbol trap for hand-written CUDA-C is deferred because inline CUDA-C is banned
in v1 (Triton-only).

Scope note: the denylist targets HIGH-LEVEL fused/compute ops (delegation). Reconstructing a
kernel from eager primitives (`mean`/`pow`/`sqrt`/`mul`) is not caught here — but it is caught
by the static guard's `require_triton` check, and such a kernel loses on the speed axis anyway.

Run during a VALIDATION pass (not the timed reps — the modes add per-op overhead). The
canonical denylists move to cco.config.json (Step 12); defaults below mirror the static guard.

Usage (needs torch; run in the WSL env):
    ~/cco-gpu/bin/python cco/dispatch_trap.py --self-test
"""

from __future__ import annotations

import contextlib
import sys


class DelegationError(RuntimeError):
    """Raised when kernel_fn executes a banned high-level/compute op at runtime."""


# High-level public function / method names (the TorchFunctionMode layer).
DENIED_FN_NAMES = frozenset({
    "matmul", "mm", "bmm", "addmm", "addbmm", "baddbmm", "mv", "dot", "vdot",
    "inner", "outer", "ger", "tensordot", "einsum", "kron", "chain_matmul",
    "linear", "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "scaled_dot_product_attention",
    "softmax", "log_softmax", "layer_norm", "rms_norm", "group_norm",
    "silu", "glu",
    "__matmul__", "__imatmul__",
})

# Base ATen op names, namespace stripped (the TorchDispatchMode layer).
DENIED_ATEN_OPS = frozenset({
    "mm", "matmul", "bmm", "addmm", "addbmm", "baddbmm", "_addmm_activation",
    "mv", "dot", "vdot", "inner", "outer", "ger", "tensordot", "einsum",
    "linear", "_linear",
    "convolution", "_convolution", "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "cudnn_convolution", "cudnn_convolution_transpose",
    "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_attention_math",
    "_flash_attention_forward", "_efficient_attention_forward",
    "softmax", "_softmax", "_safe_softmax", "log_softmax", "_log_softmax",
    "layer_norm", "native_layer_norm", "rms_norm", "_fused_rms_norm",
    "group_norm", "native_group_norm",
    "silu", "silu_", "glu",
})


def denylists_from_config(config_path: str):
    """Return (denied_fn_names, denied_aten_ops) frozensets from a cco.config.json runtime block
    (the canonical copy). The module defaults above must stay equal to those lists."""
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        r = json.load(f)["no_delegation"]["runtime"]
    return frozenset(r["denied_fn_names"]), frozenset(r["deny_aten_ops"])


def _op_base_name(func) -> str:
    """Return the namespace-stripped ATen op name, e.g. aten::mm -> 'mm'."""
    schema = getattr(func, "_schema", None)
    if schema is not None:
        return schema.name.split("::")[-1]
    bits = str(func).split(".")            # fallback: 'aten.mm.default' -> 'mm'
    return bits[1] if len(bits) > 1 else bits[0]


def _run_under_traps(kernel_fn, inputs: dict, denied_fns, denied_aten, raise_on_hit):
    """Execute kernel_fn(**inputs) under the function-level + dispatch-level traps.

    Returns (output, hits). With raise_on_hit=True, raises DelegationError on the first hit.
    """
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    hits: list[str] = []

    def record(label: str):
        hits.append(label)
        if raise_on_hit:
            raise DelegationError(
                f"kernel_fn invoked banned op {label} at runtime "
                f"(delegation to a high-level/vendor op is not allowed)"
            )

    class _FnMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if getattr(func, "__name__", "") in denied_fns:
                record(f"torch:{func.__name__}")
            return func(*args, **kwargs)

    class _DispMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            base = _op_base_name(func)
            if base in denied_aten:
                record(f"aten::{base}")
            return func(*args, **kwargs)

    with _DispMode(), _FnMode():
        out = kernel_fn(**inputs)
    return out, hits


@contextlib.contextmanager
def delegation_trap(denied_fns=DENIED_FN_NAMES, denied_aten=DENIED_ATEN_OPS):
    """Context manager: any banned high-level/aten op executed INSIDE raises DelegationError.

    Wrap a whole loop (warmup + timed reps + post-validation), not just single validation calls,
    so EVERY kernel invocation is trapped. Otherwise a kernel can probe whether it is currently
    under the trap (catch DelegationError) and delegate to a fast vendor op only in an untrapped
    phase (e.g. the timed loop), winning on the delegated kernel's latency. The mode classes and
    the (frozenset) denylists are bound here, out of the submission's reach.
    """
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class _FnMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if getattr(func, "__name__", "") in denied_fns:
                raise DelegationError(
                    f"kernel invoked banned op torch:{func.__name__} at runtime "
                    f"(delegation to a high-level/vendor op is not allowed)")
            return func(*args, **kwargs)

    class _DispMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            base = _op_base_name(func)
            if base in denied_aten:
                raise DelegationError(f"kernel invoked banned op aten::{base} at runtime")
            return func(*args, **kwargs)

    with _DispMode(), _FnMode():
        yield


def run_guarded(kernel_fn, inputs: dict,
                denied_fns=DENIED_FN_NAMES, denied_aten=DENIED_ATEN_OPS):
    """Call kernel_fn(**inputs) under the trap; raise DelegationError on the first banned op.
    Returns kernel_fn's output if clean."""
    with delegation_trap(denied_fns, denied_aten):
        return kernel_fn(**inputs)


def collect_delegations(kernel_fn, inputs: dict,
                        denied_fns=DENIED_FN_NAMES, denied_aten=DENIED_ATEN_OPS):
    """Call kernel_fn(**inputs) WITHOUT raising; return (output, hits). Empty hits == clean."""
    return _run_under_traps(kernel_fn, inputs, denied_fns, denied_aten, raise_on_hit=False)


# --------------------------------------------------------------------------------------
# Self-test (needs torch; runs on CPU tensors — no GPU required)
# --------------------------------------------------------------------------------------

def _self_test() -> int:
    import torch

    a = torch.randn(8, 8)
    b = torch.randn(8, 8)
    x = torch.randn(8, 16)

    def legit_alloc_and_elementwise(a, b):           # allocation + primitives + view: allowed
        y = torch.empty_like(a)
        y.copy_(a)
        y.add_(b)
        return y.reshape(a.shape)

    def cheat_matmul(a, b):
        return torch.matmul(a, b)

    def cheat_method_mm(a, b):
        return a.mm(b)

    def cheat_at_operator(a, b):
        return a @ b

    def cheat_addmm(a, b):
        return torch.addmm(a, a, b)

    def cheat_silu(x):
        return torch.nn.functional.silu(x)

    def cheat_softmax(x):
        return torch.softmax(x, dim=-1)

    def cheat_sdpa(x):
        q = x.reshape(1, 1, x.shape[0], x.shape[1])
        return torch.nn.functional.scaled_dot_product_attention(q, q, q)

    cases = [
        ("legit alloc + elementwise", legit_alloc_and_elementwise, {"a": a, "b": b}, False),
        ("torch.matmul",              cheat_matmul,                {"a": a, "b": b}, True),
        ("tensor .mm()",              cheat_method_mm,             {"a": a, "b": b}, True),
        ("@ operator",                cheat_at_operator,           {"a": a, "b": b}, True),
        ("torch.addmm",               cheat_addmm,                 {"a": a, "b": b}, True),
        ("F.silu",                    cheat_silu,                  {"x": x},         True),
        ("torch.softmax",             cheat_softmax,               {"x": x},         True),
        ("F.scaled_dot_product_attn", cheat_sdpa,                  {"x": x},         True),
    ]

    failures = 0
    for label, fn, inputs, expect in cases:
        _, hits = collect_delegations(fn, inputs)
        got = bool(hits)
        if got == expect:
            print(f"ok    {label:30s} -> {('caught ' + str(sorted(set(hits)))) if got else 'clean'}")
        else:
            failures += 1
            print(f"FAIL  {label:30s} -> expected delegation={expect}, got hits={hits}")

    try:
        run_guarded(cheat_matmul, {"a": a, "b": b})
        failures += 1
        print("FAIL  run_guarded did not raise on torch.matmul")
    except DelegationError:
        print("ok    run_guarded raises DelegationError on torch.matmul")
    out = run_guarded(legit_alloc_and_elementwise, {"a": a, "b": b})
    if out is not None and out.shape == a.shape:
        print("ok    run_guarded returns output for a legit kernel")
    else:
        failures += 1
        print("FAIL  run_guarded mishandled a legit kernel")

    print("-" * 60)
    print("SELF-TEST PASSED" if not failures else f"SELF-TEST FAILED: {failures} case(s)")
    return 1 if failures else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Runtime no-delegation trap for CCO submissions.")
    p.add_argument("--self-test", action="store_true", help="run built-in test cases (needs torch)")
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    p.error("nothing to do; pass --self-test (or import run_guarded/collect_delegations)")


if __name__ == "__main__":
    sys.exit(main())
