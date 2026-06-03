# Contributing to CCO

Thanks for your interest. CCO's value compounds in proportion to how many people contribute kernels, benchmark numbers, and KB entries — every PR that improves any of those is welcome.

This document is the contributor's reference. It is intentionally explicit. New contributors should be able to read it once and know what an acceptable PR looks like.

---

## Contents

1. [Ways to contribute](#ways-to-contribute)
2. [Development setup](#development-setup)
3. [Adding a new kernel](#adding-a-new-kernel)
4. [Submitting benchmark results](#submitting-benchmark-results)
5. [Knowledge-base contributions](#knowledge-base-contributions)
6. [Reflexion log format](#reflexion-log-format)
7. [What `kernel.py` may not do](#what-kernelpy-may-not-do)
8. [Code style & PR conventions](#code-style--pr-conventions)
9. [Reporting bugs](#reporting-bugs)
10. [Security](#security)
11. [License](#license)

---

## Ways to contribute

The framework's value compounds across several axes. Pick the one that fits.

### With an NVIDIA GPU

- **Optimize a bundled kernel** — pick one from `kernels/`, run the protocol, land a `kernels_optimized/<name>.py`. The most direct path to growing `CUDA_OPTIMIZATION.md`.
- **Submit a benchmark row** — fill in a `_TBD_` in `BENCHMARKS.md` on your hardware. See [#5](https://github.com/zeokin/Cuda-OSS/issues/5), [#17](https://github.com/zeokin/Cuda-OSS/issues/17).
- **Add a new bundled kernel** — propose first in an issue (label `type:kernel-request`), then ship the four-piece package (baseline + reference + config + bench-passing).
- **Add hardware-tier support** — wire roofline numbers for an unsupported GPU into `tools/bench.py`. Label `type:hardware-support`.

### Without a GPU

- **Documentation** — every doc file in this repo is contributor-improvable. The four references in [`docs/`](docs/) are especially valuable to keep accurate.
- **Issue triage** — reproduce bugs, narrow scope, close stale issues, identify duplicates.
- **KB review** — read `CUDA_OPTIMIZATION.md` and `memory/<kernel_type>.md` files; flag claims that aren't well supported by the linked experiment data.
- **Walkthroughs** — write "your first contribution" guides under `docs/` for specific contributor profiles.
- **Code review** — review open PRs, especially benchmark submissions and kernel additions.

### Tooling

- `tools/bench.py`, `tools/ncu_profile.py`, `tools/run_loop.py`, `tools/prepare.py`, `tools/merge_results.py` — all open to improvement. Label `type:tooling`. The constraint: tools must emit greppable `key=value` output that an LLM agent can parse without prose.

---

## Development setup

```bash
git clone https://github.com/zeokin/Cuda-OSS.git
cd Cuda-OSS

# Install with dev extras
uv sync --extra dev

# Validate environment (skips GPU-dependent checks if no CUDA)
uv run tools/prepare.py
```

You will need, for the full loop:

- An NVIDIA GPU from [the targeted list](README.md#targeted-hardware)
- CUDA Toolkit matching your driver
- `ncu` (Nsight Compute CLI) on `PATH` for profiling contributions
- Python 3.10+

For docs-only / triage / review contributions: just a checkout and Python. No GPU required.

---

## Adding a new kernel

A kernel contribution needs four pieces in three places. Use an existing kernel (e.g. `rms_norm`) as the canonical example.

### 1. Baseline kernel — `kernels/<name>.py`

Must export:

```python
KERNEL_TYPE: str          # e.g. "rms_norm" — must match a kernel_configs/<name>.* pair

def kernel_fn(**inputs) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """The kernel under optimization. The agent edits this file."""

def get_inputs() -> dict:
    """Return one sample input dict for smoke tests."""

def get_flops() -> int:
    """Total FLOPs at the smoke-test size, for roofline."""

def get_bytes() -> int:
    """Total bytes read+written at the smoke-test size, for roofline."""
```

The baseline must be a **working but un-optimized** implementation. Readability beats cleverness. The point is to have a clean "before" snapshot against which optimized versions are diffed.

For CUDA C kernels: ship a sibling `kernels/<name>.cu` and have `kernels/<name>.py` compile it via `torch.utils.cpp_extension.load_inline()`.

### 2. Reference implementation — `references/<name>.py`

Pure PyTorch. No Triton. No custom CUDA. Used by the bench harness as the correctness oracle.

The simplest, most obviously-correct implementation wins. If `torch.nn.functional.rms_norm` exists, use it. The reference must be readable end-to-end in under a minute by anyone familiar with PyTorch — that is its job.

### 3. Benchmark config — `kernel_configs/<name>.toml` + `kernel_configs/<name>.py`

The TOML declares sizes, dtypes, tolerances. Copy the annotated example at
`kernel_configs/SCHEMA.toml` as your template — all fields are documented inline.

```toml
[meta]
multi_output = false
test_dtypes = ["float16", "float32", "bfloat16"]

[[test_sizes]]
label = "tiny"
params = { M = 128, N = 256 }

[tolerances.float16]
atol = 1e-2
rtol = 1e-2

[[edge_sizes]]
label = "edge_1023"
params = { M = 1023, N = 256 }
```

See `kernel_configs/SCHEMA.toml` for the full reference with all fields documented.
```

The companion `.py` provides callables matching the contract `kernel_configs/_utils.py` expects:

```python
def input_generator(shape: dict) -> dict: ...
def reference_fn(**inputs) -> torch.Tensor | tuple[torch.Tensor, ...]: ...
def flops_fn(shape: dict) -> int: ...
def bytes_fn(shape: dict) -> int: ...
```

Both files are auto-discovered by `tools/bench.py` at import time — no central registry to update.

### 4. Validate end-to-end

```bash
cp kernels/<name>.py kernel.py
uv run tools/bench.py          # full pipeline must pass (correctness PASS, all 5 stages)
uv run tools/bench.py --quick  # quick pipeline must pass
uv run tools/ncu_profile.py    # NCU must produce a valid report
```

Open the PR only after all three commands succeed on at least one targeted GPU. Attach the output of `tools/prepare.py` so reviewers can match your environment.

---

## Submitting benchmark results

Benchmark rows in `BENCHMARKS.md` are the public-facing evidence of CCO's claims. We will not publish a row whose numbers we cannot reproduce.

### Prerequisites

- The kernel must already exist in `kernels/` and `kernels_optimized/`
- Numbers must come from `tools/bench.py`, not a custom harness
- You must have run the **full optimization loop** (multiple experiments through `tools/run_loop.py`), not a one-shot edit

### Required content in the PR

Use this template in the PR body:

```markdown
## Benchmark submission

- **Kernel:** `rms_norm`
- **GPU:** NVIDIA H100 80GB HBM3
- **Driver:** 550.54.15
- **CUDA:** 12.4
- **Triton:** 3.1.0
- **Agent + model:** Claude Opus 4.7
- **Final git SHA:** `<sha>`
- **Accepted iterations:** 14
- **Baseline (ms):** 0.612
- **Optimized (ms):** 0.341
- **Speedup:** 1.79×
- **% of peak (BW or FLOPs as relevant):** 87.4% BW
- **Token cost (input + output):** 412k / 38k
```

Attach `workspace/results.tsv` to the PR (or paste the last 10 rows inline). Attach the relevant `memory/<kernel_type>.md` so reviewers can read the reasoning trail.

PRs that change a row without attached artifacts will be asked for them before merge. PRs that change a row downward (regression) require a separate `type:perf-regression` issue first.

---

## Knowledge-base contributions

`CUDA_OPTIMIZATION.md` is the project's long-term artifact. It is maintained primarily by the agent during runs, but human PRs that refine or correct entries are welcome.

### Format

The file has two tiers:

**Per-kernel section** — one section per kernel type. Each contains:

- A short *Characteristics* block (bottleneck class, data access pattern, typical sizes).
- *Effective Optimizations* — numbered list, each with the technique, why it works, expected speedup range, and tagged bottleneck categories in square brackets (`[register-pressure]`, `[occupancy]`, `[cache]`, …).
- *Anti-patterns* — things that were tried and failed, with the observed regression.

**Cross-Kernel Optimization Patterns section** — indexed by bottleneck tag, not by kernel. Promote a pattern here when it has been confirmed across multiple kernel types.

### Rules for an entry

A KB entry should:

- Cite the source kernel and link the originating experiment row in `workspace/results.tsv` (or the relevant `memory/<kernel_type>.md` block).
- Quote concrete numbers (`+40% latency`, `register count 96 → 39`, `L1 hit rate 0% → 32%`), not adjectives ("significant improvement").
- Identify the *bottleneck* the technique addresses, not just the *change*.
- For anti-patterns: explain *why* it failed, with NCU evidence where available.

### When to promote to "Cross-Kernel Patterns"

A pattern earns promotion when it has been observed and confirmed in **at least three accepted experiments across at least two distinct kernel types**, with consistent direction of effect. Patterns that work on one kernel and backfire on another stay in the per-kernel sections (often as anti-patterns for the second kernel).

### What does NOT belong in the KB

- One-off shape-specific tunings (`BLOCK_SIZE_M=128 for M=4096, N=5120 on H100` is per-config, not pattern-level).
- Speculation. Every claim should be backed by an experiment row.
- Code samples. The KB describes *what* and *why*; the kernel source is *how*.

---

## Reflexion log format

After each experiment (kept *or* reverted), append a structured Reflexion block to `memory/<kernel_type>.md`. This captures the *reasoning* that the bare results.tsv row cannot — the predicted-vs-actual delta, whether the diagnosis was correct, and what should be believed going forward.

The format (also documented in [#22](https://github.com/zeokin/Cuda-OSS/issues/22)):

```markdown
### Reflexion — <experiment_id>
- **Diagnosed:** <1-sentence root cause grounded in NCU + macro analysis>
- **Hypothesis:** <what change was made and why it was expected to help>
- **Expected delta:** <e.g. "+5–10% throughput" / "reduce L2 miss rate by half">
- **Actual delta:** <measured outcome from bench.py>
- **Diagnosis correct?** yes / partial / no — <one phrase>
- **Lesson:** <1–2 sentences future runs should know>
```

When the same lesson appears in ≥3 reflexion blocks across ≥2 distinct kernel types, promote it to `CUDA_OPTIMIZATION.md § Cross-Kernel Optimization Patterns` with the appropriate bottleneck tag.

The Reflexion log is what turns the project's data into knowledge. Skipping it is the single fastest way to break the KB-as-artifact bet.

---

## What `kernel.py` may not do

The framework optimizes *the kernel you write*, not your ability to delegate to a library. A `kernel_fn` that calls back to PyTorch for the computation is forbidden, even though it would pass correctness.

### Forbidden in the body of `kernel_fn` (and any helper it calls)

- `torch.nn.functional.rms_norm`, `torch.nn.functional.layer_norm`
- `torch.nn.functional.scaled_dot_product_attention`
- `torch.nn.functional.silu`, `torch.nn.functional.softmax`
- `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.einsum`
- `torch.ops.aten.*` for the same ops
- Anything else that re-implements the kernel via a vendor or framework call

### Permitted

- Tensor allocation: `torch.empty`, `torch.zeros`, `torch.ones`
- Reshape / view / transpose: `.view()`, `.reshape()`, `.transpose()`, `.contiguous()`
- Dtype casts: `.to(dtype)`, `.float()`, `.half()`
- Shape introspection: `.shape`, `.stride()`, `.dim()`

The rule exists because a delegated `kernel_fn` would pass the 5-stage correctness check, report whatever roofline percentage the delegated op happens to hit, look like a legitimate optimization, and then poison `CUDA_OPTIMIZATION.md` with a "lesson" that has no relation to any kernel code we actually wrote. The KB is the artifact. The KB cannot tolerate this.

[#21](https://github.com/zeokin/Cuda-OSS/issues/21) tracks the AST-level guard in `tools/bench.py`. Until that lands, the rule is enforced by review — reviewers will reject any kernel that delegates.

---

## Code style & PR conventions

### Style

- Python ≥ 3.10. Type hints encouraged but not required inside kernel code.
- `ruff` for lint — run `uv run ruff check .` locally. CI runs `ruff check .` directly (no `uv sync`) so the lint job stays under a second. Line length 120, see `pyproject.toml`.
- No new top-level runtime dependencies without prior discussion in an issue. The framework's small surface is a feature.
- Tools must emit **greppable `key=value` output**, not prose. An LLM agent parses this; nothing should require regex over English.

### PRs

- One logical change per PR.
- Title format: `feat(kernel): add foo_kernel` / `fix(bench): handle dtype mismatch` / `docs: …` / `chore: …`.
- Reference any related issue in the description.
- For kernel additions, include a short paragraph on what the kernel computes and where it's used (which model family, which layer, paper if relevant).
- For benchmark submissions, include the template in [Submitting benchmark results](#submitting-benchmark-results).
- For doc/KB changes, link to the experiment(s) that motivated the change.

### Things that will get a PR sent back

- Adding a new top-level dependency without discussion
- Touching `references/<name>.py` without a correctness-related reason
- Adding code to `tools/` that imports a kernel — tools should be kernel-agnostic
- KB entries with no experiment data behind them
- A `kernels_optimized/<name>.py` that does not pass the 5 correctness stages on at least one targeted GPU

---

## Reporting bugs

Open an issue with:

- GPU model, driver, CUDA version (paste `nvidia-smi` and `nvcc --version`)
- The exact command that triggered the bug
- The full error output (do not summarize — the stack trace matters)
- A minimal repro if you can produce one
- Whether `tools/prepare.py` passes; if not, paste its output

For correctness bugs in a kernel: also include the input shapes and dtypes that triggered the discrepancy, and the `pct_within_tol` figure from the bench output.

Use these labels where possible: `type:bug`, `type:correctness`, `type:tooling`, `type:perf-regression`. Maintainers will assign priority (`P0…P3`).

---

## Security

Do not file security issues in the public tracker. Email the maintainer privately first. The `gh` token in any repo's git history is a security issue and must be revoked, not just rotated.

---

## License

By contributing you agree that your contribution is licensed under the MIT License (see [LICENSE](LICENSE)). The MIT license is non-negotiable — CCO depends on it staying permissive so optimized kernels can ship into closed-source production.
