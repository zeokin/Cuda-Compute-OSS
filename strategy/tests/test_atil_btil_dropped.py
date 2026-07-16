"""CPU-only tests: multiply_subspace drops Atil/Btil before reconstruct.

After Ctil = Atil @ Btil, the two (m, m) compress results are unused. Keeping
them live through reconstruct under-budgets by 2*m*m on MPS (static free
ceiling) and can OOM. Pure source/arithmetic checks; no GPU needed.

Run:  python strategy/tests/test_atil_btil_dropped.py
"""
from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy import subspace


def test_multiply_subspace_deletes_atil_btil_before_reconstruct():
    src = inspect.getsource(subspace.multiply_subspace)
    del_at = src.index("del Atil, Btil")
    recon_at = src.index("reconstruct(")
    assert del_at < recon_at, "Atil/Btil must be dropped before reconstruct runs"


def test_leftover_atil_btil_would_overshoot_reconstruct_budget():
    """Arithmetic witness: reconstruct's charged peak fits; adding 2*m*m leftovers
    that stay live through the stream overshoots (the pre-fix bug)."""
    # Regime where Q+Ctil alone fit the frac budget (unlike M=N at tiny free).
    n, m, item, frac = 4096, 2048, 4, 0.3
    free = 300 * 1024**2
    budget = int(free * frac)
    fixed = (n * m + m * m) * item          # Q + Ctil (what reconstruct charges)
    per_row = (n + m) * item
    blk = max(1, (budget - fixed) // per_row)
    peak = fixed + blk * per_row
    assert peak <= budget

    leftover = 2 * m * m * item             # Atil + Btil if kept alive
    assert peak + leftover > budget
    assert (peak + leftover) / budget > 1.3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
