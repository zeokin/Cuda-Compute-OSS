"""Subspace strategy examples. Run:  python strategy/examples/run_example.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import subspace_matmul, Config, Transform, register_transform

try:
    import torch
    _gpu = torch.cuda.is_available() or (getattr(torch.backends, "mps", None)
                                         and torch.backends.mps.is_available())
except Exception:  # noqa: BLE001
    _gpu = False
if not _gpu:
    sys.exit("This example needs a GPU (CUDA/MPS) — CCO computes on the GPU only.")


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# 1) Smart multiply on LOW-RANK data: near-exact at a fraction of the FLOPs.
r = 16
A = (np.random.rand(1024, r) @ np.random.rand(r, 1024)).astype("float32")
B = (np.random.rand(1024, r) @ np.random.rand(r, 1024)).astype("float32")
C = subspace_matmul(A, B, config=Config(transform="rsvd", rank_m=128, verbose=False))
print("1) rsvd(M=128) on rank-16 data, rel err:", rel(C, A @ B))

# 2) The honest failure case: full-rank random data -> large error.
Af = np.random.rand(512, 512).astype("float32")
Bf = np.random.rand(512, 512).astype("float32")
Cf = subspace_matmul(Af, Bf, config=Config(transform="rsvd", rank_m=64, verbose=False))
print("2) rsvd(M=64) on FULL-rank data, rel err:", rel(Cf, Af @ Bf), "(expected: large)")

# 3) Plug in your own transform (the updatable core tech).
class FirstAxes(Transform):
    name = "firstaxes"
    def basis(self, n, m, backend, dtype, A=None, B=None):
        Q = np.zeros((n, m), dtype=dtype)
        Q[np.arange(m), np.arange(m)] = 1.0
        return backend.to_device(Q)

register_transform("firstaxes", FirstAxes)
C3 = subspace_matmul(A, B, config=Config(transform="firstaxes", rank_m=1024, verbose=False))
print("3) custom transform (M=N) rel err:", rel(C3, A @ B))
