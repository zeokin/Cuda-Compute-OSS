# Evaluation packages

`eval/` contains two deliberately separate systems.

## Attention foundation (current draft)

- `attention_manifest.py`: validates and hashes the protected workload.
- `attention_benchmark.py`: emits raw RTX 5070 Ti correctness/timing artifacts.
- `attention_decision.py`: compares current main with one PR artifact.
- `attention_batch.py`: previews, evaluates, and—only after activation—processes
  the sequential PR queue.

The manifest is currently `draft`; processing and automatic merging are locked.
Public status and contribution scope live in the root README.

## Square GEMM (legacy)

The remaining evaluator, tracks, ledger, GPU batch runner, and result bot are
retained to reproduce historical square-matrix results. They do not establish
attention or model improvements and are not the active miner lane.
