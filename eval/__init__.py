"""Evaluation system for the smart (subspace) matrix-multiplication strategy.

Generates random matrix couples, multiplies them with **normal** (exact) and
**smart** (subspace) computing, and scores each transform strategy on accuracy,
latency, and peak VRAM:

    Accuracy = max(0, 1 - ||C - Ĉ||_F / ||C||_F)
    score    = Accuracy × (1 / Peak_VRAM) × (1 / Latency)   (0 if accuracy < floor)

Quick API
---------
    from eval import EvalConfig, evaluate
    out = evaluate(EvalConfig(n=12000, pairs=3))
    print(out["best"], out["ranking"])

Or from the CLI:  ``python -m eval --n 12000 --pairs 3``
"""
from __future__ import annotations

from . import metrics, memory
from .evaluator import EvalConfig, evaluate, estimate_scaling

__all__ = ["EvalConfig", "evaluate", "estimate_scaling", "metrics", "memory"]
