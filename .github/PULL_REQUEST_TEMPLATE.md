<!--
CCO accepts only measurable feature PRs for the phase currently marked ACTIVE
in README.md and the protected benchmark manifest. If the phase is BUILDING or
CALIBRATING, ordinary miner PRs are closed because their evaluator is not yet
frozen. Bug-fix, docs, cleanup, and legacy square-transform PRs are not mining
lanes.
-->

## Current-phase declaration

- [ ] feature

**Benchmark:** `attention-foundation-v1-rtx5070ti`

Closes #____

<!-- The issue must be an approved current-phase issue opened before this PR. -->

## Measurable improvement

<!-- Explain the implementation change and why it should improve an official workload. -->

## Scope

- [ ] prefill
- [ ] decode
- [ ] both

<!-- One focused implementation change under attention/, matmul/, and tests/. -->

## Correctness and validation

<!-- List exact commands and results. Do not paste a legacy square-GEMM score. -->

## Risks and non-goals

<!-- State unsupported shapes, layouts, dtypes, or performance trade-offs. -->

## Checklist

- [ ] This PR implements an approved current-phase issue.
- [ ] This is a measurable feature, not a fix, documentation, cleanup, or refactor-only PR.
- [ ] It does not modify `eval/`, `benchmarks/`, `.github/`, `dashboard/`, public policy, or scoring rules.
- [ ] Exact output, dtype, shape, causal, finite-output, and relevant edge-case tests pass.
- [ ] The full CPU-safe test suite passes.
- [ ] My commits do not credit a coding agent in `Co-authored-by` footers.
- [ ] I understand that contributor measurements are diagnostic; the protected RTX 5070 Ti evaluator decides the result.
