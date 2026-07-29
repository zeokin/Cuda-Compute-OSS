---
name: Current-phase feature proposal
about: Propose one measurable implementation improvement for the active benchmark
title: "[feature] "
labels: "type:feature,status:triage"
assignees: ""
---

## Benchmark phase

`attention-foundation-v1.1-rtx5070ti`

## Workload affected

- [ ] prefill
- [ ] decode
- [ ] both

## Current limitation

<!-- Identify the measured implementation bottleneck or missing exact primitive. -->

## Proposed implementation

<!-- Keep this to one independently measurable feature. -->

## Correctness risks

<!-- Address shape, dtype, causality, finite outputs, layouts, and ragged inputs. -->

## Expected measurement

<!-- Explain what official workload should improve and what must not regress. -->

## Scope

Expected implementation paths: `attention/`, `matmul/`, and focused `tests/`.
Do not implement until a maintainer confirms that the benchmark phase is ACTIVE
and applies `status:phase-approved` to this issue.
