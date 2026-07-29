# Evaluation packages

`eval/` contains two deliberately separate systems.

## Attention foundation (active)

- `attention_manifest.py`: validates and hashes the protected workload.
- `attention_benchmark.py`: emits raw RTX 5070 Ti correctness/timing artifacts.
- `attention_decision.py`: compares current main with one PR artifact.
- `attention_batch.py`: previews, evaluates, and processes the sequential PR
  queue under the frozen benchmark and explicit confirmation lock.

The v1.1 manifest is calibrated, certified, frozen, and active. Public status
and contribution scope live in the root README.

## Square GEMM (legacy)

The remaining evaluator, tracks, ledger, GPU batch runner, and result bot are
retained to reproduce historical square-matrix results. They do not establish
attention or model improvements and are not the active miner lane.
