# Testing Strategy — Local vs. Validator

*Reference plan, July 2026. Companion to [sn74-emission-strategy.md](sn74-emission-strategy.md).
Pins: **GPU = RTX 5070 Ti** · **reference model = Gemma 4 E4B**.*

## Two tiers

**Tier 1 — local (miner-side, informational, never scoring).**
CPU or GPU per the miner's own hardware/demand. Always includes a smoke test:
tiny N, must finish in seconds, checks shapes/orthonormality/no-NaNs — a
pre-flight, not a measurement. Miners may also self-run the real
`python -m eval` at whatever scale their hardware allows, purely to iterate;
these numbers are never binding.

**Tier 2 — online/validator (authoritative, produces the score).**
Runs only on the pinned RTX 5070 Ti, on pinned workload shapes, and is what
`eval/evaluator.py` already does structurally: compute the exact product
**once**, then run every submitted strategy on that **identical input**,
reporting accuracy, latency, peak VRAM, and FLOP ratio side by side, gated by
the dominance rule (`metrics.py`). The bot automates this per PR; nothing new
needs inventing at the algorithm level, only hardening:

- **Fresh random seed per validation run** (today's `EvalConfig.seed` defaults
  to 0 — a known overfitting exploit; a submission could hardcode the answer
  for a fully-known fixed input). Log the seed with every result so the exact
  run is reproducible.
- **Vary the regime, not just the seed.** Run each submission across multiple
  data tracks (full-rank, low-rank, decaying-spectrum, and the Gemma-4-derived
  track below) so a submission can't overfit one structural assumption even if
  it can't predict one seed.
- **Both baseline and candidate, same input, every axis at once** — already
  implemented; the dominance gate refuses to score anything that isn't better
  on accuracy, latency, VRAM, and FLOPs simultaneously.

## Pinned hardware — RTX 5070 Ti

Chosen for developer-tier accessibility and proven availability in cloud
providers. Self-hosted on a Windows box for reliability and consistent CI/CD
workflow. All official measurements and verdicts are pinned to RTX 5070 Ti;
absolute latency and VRAM numbers are device-specific and should not be
cross-compared with historical results from other devices (e.g., RTX PRO 500
ledger entries are historical reference only).

## Reference model — Gemma 4 E4B

Real, shipped April 2, 2026 (~4.5B effective params, Apache 2.0, built from
Gemini 3 research). Chosen specifically because its architecture already
matches the hybrid operator this project is building toward: **5 local
sliding-window layers (512-token window) + 1 global layer, repeated 7×, final
layer always global.** The spectral hybrid mixer's job is to replace that
global layer — E4B is not just "a model to test on," it's the same shape as
whitepaper Figure 4.

How to use it at the current stage (matmul simulation, not yet full attention):

1. **Don't run full inference yet.** Pull the real `config.json` from
   `google/gemma-4-E4B` on Hugging Face and use its actual d_model, head
   count, and 512-token window to fix evaluation shapes — never hardcode
   guessed numbers.
2. **Add a Gemma-4-derived data track.** Generate one test regime from Gemma
   4 E4B's real trained weight matrices instead of purely synthetic data.
   Trained weights are usually not full-rank, so this sits as a genuinely
   useful middle regime between synthetic tracks and the eventual full
   attention-shaped benchmark — the concrete bridge from where the repo is
   today to the spectral-mixing milestones (M1–M4).

## Open code TODOs this implies

- Fix `EvalConfig.seed` default / plumb a fresh seed per validator run
  (`eval/evaluator.py`).
- Add `low-rank`, `decaying-spectrum`, and `gemma4-weights` as selectable
  `fill` regimes alongside today's `random`/`lowrank`/`iota`.
- Script the Tier-1 smoke test as a single command building on the existing
  GPU-optional skip behavior in `tests/test_correctness.py` and
  `strategy/tests/test_subspace.py`.

Not yet implemented — flagging for a future session when ready to build.
