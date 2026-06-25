# CCO — Design, Scoring & Threat Model

How the competition is built, how a submission is scored, and why it's hard to cheat. For the
miner-facing rules see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 1. The split: locked substrate vs. variable artifact

The single most important design decision: **the runtime is fixed and shared; the submission is the
only variable, and it is fed in.**

- **Locked (byte-verified at every PR HEAD against `manifest.json`):** the harness `benchmark.py`, the
  correctness oracles `references/`, the benchmark spec + input generation `kernel_configs/`, the
  per-track champions `champions/`, the enforcement code `cco/`, the config, and the runtime image.
- **Variable:** exactly one file, `kernel.py` (`KERNEL_TYPE` + `kernel_fn`), bound by `kernel_sha256`.
- **Per-PR secret:** the input seed = a function of the PR HEAD SHA (`cco/seed.py`), unknowable to
  the miner in advance.

The automated gate pipeline runs off-repo against this locked substrate; this repo ships only the
substrate + the per-repo config/state it reads.

## 2. The gate walk (default verdict = reject)

A PR merges only on affirmative evidence. The gate pipeline short-circuits on the first failure:

1. **Parse** — the fenced JSON payload against `payload-schema.json`; malformed → reject.
2. **Gate 1 — identity** — GitHub ↔ hotkey ↔ hotkey on the SN74 metagraph ↔ payload
   signature verifies under the hotkey.
3. **Gate 2 — manifest integrity** — re-hash every path in `main:manifest.json` at the PR HEAD; only
   `kernel.py` may differ, and **no unlisted file may be added** to a locked directory
   (`cco/manifest_tool.py` pins the full directory listing, closing the `kernel_configs` auto-import
   RCE). Gate 2 additionally rejects any PR whose **git diff touches anything other than
   `kernel.py`** — this complementary diff rule is what covers `manifest.json` itself, `.github/`,
   the docs, and stray top-level files outside any locked path. The repo's own CI status is
   advisory; no gate consults it.
4. **Gate 3 — no-delegation static scan** — `cco/guard_kernel.py` AST-rejects high-level/vendor ops,
   the `@` operator, dynamic-dispatch escapes, inline CUDA-C, and `get_*` exports; requires a
   `@triton.jit` kernel. Cheap; runs before any GPU spend.
5. **Rate-limit** — 1 canonical rerun / hotkey / 24h.
6. **Gate 4 — canonical rerun** — on trusted GPU hardware, egress closed, on PR-HEAD-seeded inputs:
   run the 5-stage correctness gate + the scored latency sample, with the runtime no-delegation trap
   (`cco/dispatch_trap.py`) active, and emit the bound score blob.

The PR is **frozen** between gate-walk and merge (snapshot of head SHA + body hash); any drift closes
it. This removes the "pass clean, then push a backdoor before merge" window.

**Who merges and labels (king-of-the-hill).** The gate pipeline only *decides*; the merge and the
`cco-winner-<track>` label are applied by the **CCO maintainer bot** — a maintainer-owned token (an
owner/collaborator account, or a GitHub App honored via `trusted_label_pipeline`), **never** the
read-only Gittensor App (on SN74, Gittensor never merges for you). On a win the bot merges the PR,
moves the single `cco-winner-<track>` label off the prior champion onto the new one, and closes the
rest; on a loss it closes the PR. SN74 validators then only *observe* the merged + labeled state and
pay via the per-repo king-of-the-hill config (§3). Merging only the winner makes "merged" == "took the
crown."

## 3. Scoring

**Correctness is a hard gate, never an axis.** All 5 `benchmark.py` stages must PASS against the locked
oracle at the locked tolerances: smoke, shape sweep, numerical stability, **within-tolerance**
determinism (admits correct atomics/split-K kernels — not bitwise), edge cases. Speed never buys
back correctness.

**The scored axis is speedup vs the current champion — not vs the PyTorch oracle.** The oracle is
the *correctness* spec only; scoring against it would be meaningless (for `matmul` the oracle is
cuBLAS — unwinnable and a delegation magnet; for the memory-bound kernels it's slow eager PyTorch —
the first fused kernel wins 10× and the ladder then plateaus). So the bar is the standing champion
kernel, re-run **fresh and interleaved** with the challenger in the same sealed job (shared thermal
state).

**The measurement** (`run_scored_sample`): on the primary size + dtype, `n_blocks=30` block-mean
latencies — a *sample*, not a single median, because the win test is statistical. Three anti-cheat
properties are baked into the timing:
- **rotating input buffers** across reps (distinct seeds → distinct values & storage) — defeats
  warm-L2 residency and memoize-by-pointer;
- **fused correctness on 2 distinct buffers** — a kernel that caches buffer-0's answer fails on
  buffer-1 ("fast garbage at the scored size");
- an **output-vs-input alias guard** — a kernel returning a view of its input is rejected.

**The win decision** (`cco/significance.py`, run by the gate pipeline): a one-sided
**Mann-Whitney U** test on the two latency samples — nonparametric, so it's robust to the bimodal
GPU clock-boost that makes a Welch t-test misfire — **plus** an effect-size margin. A challenger
wins only if it is *significantly* faster **and** faster by ≥ `min_improvement_pct`. This rejects
both noise-only flips and statistically-detectable-but-tiny wins, preventing crown-thrash. **VRAM**
is a non-regression guard band.

**Emissions** (`cco.config.json` → `gittensor_repo_config`, mirrored into Gittensor's
`master_repositories.json`): king-of-the-hill on SN74. `fixed_base_score` + `default_label_multiplier: 0`
mean only the PR currently holding `cco-winner-<track>` earns, and the magnitude of a win doesn't change
pay — you're rewarded for *taking and holding* the frontier. SN74 validators read the **winning PR's**
label every ~2h (`trusted_label_pipeline` lets the CCO bot's label count even as a NULL-actor App;
`eligibility` is zeroed so a single crowned winner isn't gated out), and `scoring.time_decay` is
flattened so a held crown keeps paying instead of decaying by time-since-merge. The label move is
performed by the CCO bot (§2), not by Gittensor.

## 4. The bound score blob

`benchmark.py --blob` emits a sorted-keys JSON blob (`cco/blob.py`) that *proves what produced a score*:
the latency sample + correctness verdict, plus `input_seed`, `kernel_sha256` (the artifact),
`reference_hash` (oracle + config — so a score can't be computed against a weaker/edited oracle),
`harness_self_hash` (the scorer's own source), the GPU/SKU, and `blob_sha256` over all of it. Under
attestation (§6) `blob_sha256` is bound into the quote; in the v1 trusted-box posture it's the
integrity hash tying a score to its evidence. The harness makes no keep/revert decision.

**Isolation.** The authoritative scored run executes `kernel.py` in a **separate subprocess**
(`cco/isolate.py`). A kernel that shared the scorer's interpreter could otherwise read the secret
seed from `sys.argv`, monkeypatch `torch.allclose` / the timing events, or neuter the no-delegation
trap — no in-process guard can stop code that runs *inside* the interpreter. So the parent generates
the inputs from the secret seed (which **never enters the child**) and judges correctness with **its
own** oracle against the outputs the child returns — validating distinct buffers drawn both *before
and after* the timed window, so a correct-then-garbage call-counter has no safe window. The child is
launched with `-E` and a clean working directory (no `sitecustomize` / `PYTHON*` injection), and its
output is deserialized tensor-only (`weights_only=True`) so it cannot pickle-RCE the parent. The
in-child dispatch trap wraps the **entire scored window** — pre-validation, warmup, the timed loop,
and post-validation — so there is no untrapped phase in which a kernel could detect it is unobserved
(by catching `DelegationError`) and delegate to a fast vendor op only while being timed; a banned op
*anywhere* in that window is caught. The static guard's denylist is kept aligned with the runtime
trap's, and a submission may not import the `cco` package (so it cannot reach the trap internals).
Timing primitives (`torch.cuda.Event` / `synchronize` / `perf_counter`) are captured as child-locals
**before** the submission loads, and the kernel runs in a **separate worker thread** that cannot reach
them (no `time` import, no cross-thread frame-walk, no `gc` reach), so it cannot patch its own clock. The
**primary latency is the per-block synced captured-clock WALL** — `perf_counter` around each block, which
ends in a full-device `synchronize` — NOT the default-stream CUDA events: the sync makes the wall count
any work a kernel moves *off* the default stream, so a side-stream / split-K kernel that under-reports
its events gains nothing — the wall is the score and stays honest. The CUDA events are kept only as a
sanity bound (a subset of the wall; they cannot exceed it), and the absolute roofline floor is checked
against the GPU-only event time. *Residual:* the wall is measured by the (kernel-unpatchable, thread-
isolated) child clock; moving the measurement to a parent-clock interactive RPC is a documented v2
hardening, with low marginal value since the design already relies on thread isolation for the in-child
probe schedule.

**Timed-loop integrity.** The timed loop mutates one element of **every float input** before each call
(via a tiny Triton store, not a trapped torch op), so a kernel that memoizes by pointer returns a
now-stale output and one that memoizes by content must recompute (honest timing). The written value is a
**keyed hash of the call index** (server-secret), not a monotone counter, so reading it reveals no
ordinal a kernel could gate on. A server-random, kernel-unknowable sample of timed calls has its
`(mutated input, output)` captured and **re-derived against the parent's oracle** — spread across **all**
blocks so it overlaps the median-feeding calls. The input values themselves come from a **server-entropy-
salted seed** chosen at scoring time (the PR-HEAD SHA is reproducibility metadata only), so a
precompute-and-memorize attack misses.

Keeping the probe schedule **unreadable** to the kernel is load-bearing, and the structural close is
**thread isolation**: the kernel — its module-level body *and* every call — runs in a dedicated worker
thread, while the schedule, the timing events, the clock, and `job` live in the main thread. `f_back`
walks only the current thread's stack, so the schedule is off the kernel's reachable call stack; the
process-wide heap-enumeration route (`gc.get_objects`) and the module re-export / loader / `__builtins__`
routes are closed by the import allowlist + the static attribute bans (re-run inside the scoring
subprocess on the exact bytes about to execute). A schedule-aware kernel therefore cannot do real work
only where observed: it must be correct on **every** timed call or risk the probe. Speed cannot be forged
either — the **primary latency is the per-block synced captured-clock wall** (see above), so side-stream /
split-K under-reporting buys nothing, and an **absolute roofline floor** (`max(bytes/peak_bw,
flops/peak_flops)`) rejects any physically-impossible (memoized / cached) GPU time. Together these close
memoize-and-replay (incl. the padded variant), the predictable-probe-schedule evasion, the frame /
heap / re-export schedule leaks, and side-stream timing forgery. The fully process-level posture — the
kernel in a sandboxed child whose memory holds **no** secret at all — is the v2/TEE direction
([§ residuals](#)); in v1 the secret stays in the child's main thread, protected by thread isolation +
the bans, with the OS sandbox (below) containing any residual RCE.

**Native no-delegation backstop (the load-bearing guard).** The in-child dispatch trap is necessary
but not sufficient: a kernel sharing the interpreter can pop the TorchFunction/TorchDispatch mode
stack, delegate to a vendor GEMM, and push it back — all within one call, invisible to Python. So the
scoring child is launched with an **`LD_PRELOAD` shim** (`runtime/cco_preload.c`) that interposes the
vendor **compute** symbols (cuBLAS/cuBLASLt GEMM, cuDNN convolution/graph-execute, …) by name.
PyTorch links those libraries into the global symbol scope, so any `torch`→cuBLAS call — even one the
popped Python trap never saw — resolves to the shim, which records the symbol and `_exit(99)`s the
child; the parent reports it as delegation. The shim interposes *compute* entry points only (never
handle/descriptor setup), so a legit Triton kernel (which launches its own MMA via the CUDA driver)
and torch's own context init never trip it. The parent is **not** preloaded (it computes the cuBLAS
oracle), and it **refuses to score** unless a plain `torch.mm` child trips the shim — turning any
future linkage regression into a hard stop, not a silent universal bypass. *Residual:* a vendor
kernel statically compiled into `libtorch` that crosses no cuBLAS/cuDNN symbol (flash / mem-efficient
SDPA, row-wise fp8, int4-pack) is shim-blind; those remain guarded only by the static AST ban + the
(poppable) in-Python trap — hardening them is Tier-3 work.

## 5. Threat model — what's gameable, and what closes it

| Attack | Closed by | Residual? |
|---|---|---|
| Edit the harness / oracle / config / a champion | Gate 2 manifest re-hash (main-authoritative) | no |
| Inject a new file into a locked dir (auto-import RCE) | Gate 2 full-directory-listing pin | no |
| Delegate to `torch.matmul` / `F.*` / `@` / cuBLAS | static AST guard (Gate 3) **+** runtime dispatch trap (Gate 4) **+** `LD_PRELOAD` vendor-symbol trap | hand-rolled MMA vs "morally cuBLAS" is a policy line; Triton-only v1 shrinks it |
| Pop the in-Python trap mid-call, then delegate to a vendor GEMM | the `LD_PRELOAD` shim interposes the cuBLAS/cuBLASLt/cuDNN **compute** symbols in the scoring child (op-name-agnostic, unreachable from Python); harness refuses to score if it ever goes inert | fused kernels statically linked into `libtorch` (flash SDPA, row-wise fp8, int4-pack) cross no vendor symbol → guarded only by the static ban (Tier 3) |
| Delegate via an alternate GPU library (`cupy`/`jax`/`cutlass`/`numba`) or `torch.compile`/Inductor codegen | imports are an **allowlist** (only torch/triton/stdlib-utils) + the codegen namespaces (`torch.compile`/`_dynamo`/`_inductor`/`fx`/`jit`) are banned (Gate 3) | a GEMM that routes through cuBLAS is still caught by the shim; a CUTLASS kernel compiled into `libtorch` is the Tier-3 residual above |
| Dynamic dispatch to a banned op (aliasing, `getattr`/`eval`, `torch.__dict__['matmul']`) | builtins ban + introspection-dunder ban + import-alias resolution in the static guard | no (any GEMM that does execute still hits the runtime trap + the shim) |
| Walk the call stack to read the secret probe schedule (`e.__traceback__.tb_frame.f_back.f_locals`) | **THREAD ISOLATION (structural):** the submission's code — both its module-level body and every `kernel_fn` call — runs in a dedicated WORKER thread, while the probe schedule, the timing events, the captured clock, and `job` all live in the main thread. `f_back` walks only the current thread's stack, and the cross-thread frame-access escapes (`sys._current_frames`/`gc`/`threading`) are import-banned, so the schedule is unreachable regardless of HOW the kernel names an attribute (`getattr`/`operator`/`str.format`/`match`-`case`/a form not yet invented). The static attribute-name bans + Gate-4 re-scan + job-file deletion remain as defense-in-depth | the per-call main↔worker handoff adds latency that inflates the timing of *fast* kernels on a high-thread-latency host (≈2.7× on WSL2; ~0 for kernels ≳500µs since the GPU saturates and hides it). Bare-metal Linux handoffs are ~10-20µs → expected minimal; **must be measured on the production host**, and the host-independent fix (batched timing: worker runs autonomous segments, main intervenes only at the ~24 probe positions) is the fallback if it is not |
| Infer the probe schedule from the extra buffer cloned at a probed call | `torch.cuda.memory_allocated`/`memory_stats`/`mem_get_info`/… are statically banned, so the kernel cannot read allocator deltas; it also cannot time itself (no `Event`, no `time` import) | low-severity even unbanned (allocator churn is noisy); banned outright regardless |
| Under-report timing via a side CUDA stream / graph / split-K (correct work, fast default-stream events) | the **primary score is the per-block synced captured-clock WALL**, not the cuda events — the full-device sync makes the wall count work on ANY stream, so moving work off the default stream buys nothing; events are kept only as a sanity bound. (The stream/event/graph API is also statically banned under all its qualified names as defense-in-depth.) | the wall is measured by the kernel-unpatchable thread-isolated child clock; a parent-clock interactive RPC is the v2 hardening (low marginal value given thread isolation already protects the schedule) |
| RCE on the scoring host (`warnings.sys.modules['os'].system`, `mod.__loader__.exec_module`, …) — arbitrary code / file R/W | the kernel runs in an **OS sandbox** (bubblewrap: read-only root, only the per-run scratch writable, no network, dropped caps; `runtime/sandbox.sh`, opt-in via `CCO_ENABLE_SANDBOX` once host-validated) so RCE cannot touch the harness/host; the re-export / loader / `__builtins__` / string-dispatch routes are also statically banned (Gate 3 + Gate-4 re-scan) | the sandbox keeps `/dev/nvidia*` writable (CUDA needs it) so GPU-side isolation is NOT provided — bounded by no-secret-on-the-GPU + one-submission-at-a-time scheduling; GPU-TEE is v2. The static bans are an in-process denylist (non-closable in principle) — the sandbox is the structural backstop |
| Reach the scored GPU run without the static guard (Gate 3 → Gate 4 gap / TOCTOU) | the static guard is **re-scanned inside the scoring subprocess on the exact bytes about to `exec`**; any violation aborts as a delegation result before the kernel loads | no |
| Inline CUDA-C escape | banned in v1 (guard rejects `cpp_extension`) | n/a in v1 |
| Memorize / hardcode outputs for known inputs | PR-HEAD-seeded inputs the kernel **never sees** (process isolation); oracle re-derives truth | no |
| Cache first answer, return it always | parent-validates distinct buffers before + after timing | no |
| Memoize-and-replay (per-buffer cache → ~free timed loop) | per-call input mutation + kernel-unknowable timed-output probe + absolute roofline floor (§4) | no |
| Fast garbage only at the scored size | parent-validates the scored-size outputs against its oracle | no |
| Win via warm-L2 residency | rotating input buffers across reps | reduced; canonical box also locks clocks |
| Return a view of the input (no compute) | parent-side oracle validation (an unchanged input fails the oracle) | no |
| In-process score forgery (patch the comparison/timing, read the seed from `argv`) | the kernel runs **isolated** in a subprocess; the parent judges correctness with its own oracle and bounds timing by wall-clock (§4) | timing under-report (bounded) |
| Pickle-RCE the scorer from the subprocess | child output loaded `weights_only=True` (tensors only) | no |
| `os.system` / `import sys` / sitecustomize escape | static import ban (`os`/`sys`/`builtins`/`io`) + child `-E` + clean cwd | no for the verdict |
| OOM-dodge a locked size | OOM on a correctness size is a **FAIL**, not a skip | no |
| Pass intermittently / race conditions | within-tolerance determinism + multi-buffer correctness; gross races fail smoke/sweep | rare 1-in-10⁶ faults policed post-merge |
| Approximate/degraded output under loose tolerance | per-track locked tolerances (tightened; e.g. swiglu 0.5 → 0.01/0.2) | tolerance is a benchmark-validity knob; per-output tolerances are a noted refinement |
| Win on a faster GPU | the SKU is pinned + part of the locked "model" (a swap is a vN reset) | requires attested SKU for full strength (§6) |

## 6. Attestation (v1 vs v2)

A kernel competition needs a **GPU** under confidential compute, so a CPU-only TEE does not apply.
- **v1 (current posture):** the canonical rerun runs on a **trusted, pinned GPU host**, egress
  closed, clocks locked, exclusive GPU. This closes the cheating surface (CCO runs the PR's code
  itself); it defers third-party *auditability*.
- **v2:** route the rerun through GPU-attested confidential compute (a GPU TEE) so the published
  image lands in MRTD and `blob_sha256` binds into the quote — then anyone can verify the rerun was
  honest. This is the make-or-break infrastructure dependency.

## 7. What's intentionally not here

There is no in-repo optimization agent or knowledge base. CCO ships only the locked substrate and
the single mutable `kernel.py`: the optimization intelligence is the external contributors, each
submitting one artifact to a frozen, objective harness.
