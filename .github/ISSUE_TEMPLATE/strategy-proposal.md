---
name: Strategy proposal
about: Propose a new smart strategy / transform to lower matrix-multiply cost
title: "[strategy] "
labels: "type:strategy, status:triage"
---

## Idea

<!-- What subspace / transform / scheme, and why it should compress this data cheaply. -->

## Target regime

- Matrix content: <!-- low-rank / smooth / structured / … -->
- Expected `N`, `M`, dtype, device:

## Expected trade-off

Against the exact baseline, on the target regime, which axes do you expect to
move? (Per the one rule, an improvement reduces **all** cost axes with accuracy
held — see [BENCHMARKS.md](../../BENCHMARKS.md).)

| aspect          | vs exact (expect) |
|-----------------|-------------------|
| accuracy        |                   |
| time complexity |                   |
| latency         |                   |
| VRAM usage      |                   |

## Notes

<!-- References, prior art, or open questions. -->
