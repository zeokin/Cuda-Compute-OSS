"""Minimal usage examples. Run:  python examples/run_example.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul import matmul, Config
from matmul import runner

try:
    import torch
    _gpu = torch.cuda.is_available() or (getattr(torch.backends, "mps", None)
                                         and torch.backends.mps.is_available())
except Exception:  # noqa: BLE001
    _gpu = False
if not _gpu:
    sys.exit("This example needs a GPU (CUDA/MPS) — CCO computes on the GPU only.")


# 1) Simple in-memory API (computes on the GPU via PyTorch).
A = np.random.rand(2048, 2048).astype("float32")
B = np.random.rand(2048, 2048).astype("float32")
C = matmul(A, B)
print("1) matmul(2048) max abs err vs numpy:",
      float(np.max(np.abs(C - A @ B))))

# 2) Full runner with reporting (small n so verify runs).
info = runner.run(1024, Config(dtype="fp32", verbose=True), verify=True)
print("2) info:", {k: info[k] for k in ("mode", "gflops", "verify")})

# 3) How you'd launch a huge, out-of-core run (commented: needs a big GPU +
#    lots of disk). n=128000 fp32 => ~196 GB across A+B+C on disk.
#
#    from matmul import Config, runner
#    cfg = Config(dtype="fp16", storage="disk", workdir="/data/mm",
#                 vram_fraction=0.6)
#    runner.run(128000, cfg, fill="random", keep=True)
