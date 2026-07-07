"""CPU-only tests for matmul tile VRAM budgeting (no GPU required).

Run:  python tests/test_auto_tile_budget.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matmul.config import Config
from matmul.gemm import (
    _fits_in_core,
    _in_core_bytes_per_elem,
    _tile_operand_bytes,
    _tile_workspace_bytes_per_elem,
    auto_tile,
)


class _FakeBackend:
    def __init__(self, free_bytes: int):
        self._free = free_bytes

    def free_compute_bytes(self) -> int:
        return self._free


def test_tile_operand_bytes_fp16_upcast():
    fp16_acc = Config(dtype="fp16", accumulate_fp32=True)
    fp16_raw = Config(dtype="fp16", accumulate_fp32=False)
    fp32 = Config(dtype="fp32")

    assert _tile_operand_bytes(fp16_acc) == 4
    assert _tile_operand_bytes(fp16_raw) == 2
    assert _tile_operand_bytes(fp32) == 4


def test_tile_workspace_counts_fp16_upcast():
    fp16_acc = Config(dtype="fp16", accumulate_fp32=True)
    # acc fp32 (4) + two fp32 operand tiles (4 + 4)
    assert _tile_workspace_bytes_per_elem(fp16_acc) == 12

    fp16_raw = Config(dtype="fp16", accumulate_fp32=False)
    assert _tile_workspace_bytes_per_elem(fp16_raw) == 6


def _legacy_auto_tile(n: int, cfg: Config, free_bytes: int) -> int:
    """Pre-fix estimator that sized fp16 operand tiles at item_bytes."""
    budget = int(free_bytes * cfg.vram_fraction)
    per_elem = cfg.acc_dtype.itemsize + 2 * cfg.item_bytes
    t = int(math.sqrt(max(1, budget) / per_elem))
    t = min(t, n)
    t = max(128, (t // 128) * 128)
    return min(t, n)


def test_auto_tile_fp16_accumulate_smaller_than_legacy_estimator():
    """fp16+accumulate_fp32 must not inherit fp16 operand sizing in the budget."""
    free = 64 * 1024**2
    backend = _FakeBackend(free)
    n = 8192
    cfg = Config(dtype="fp16", accumulate_fp32=True, vram_fraction=0.6)

    t_fixed = auto_tile(n, cfg, backend)
    t_legacy = _legacy_auto_tile(n, cfg, free)
    assert t_fixed <= t_legacy
    assert t_fixed < t_legacy


def test_in_core_bytes_accounts_for_fp16_upcast():
    """In-core budget must count the fp32 upcasts + fp32 product, not just
    A+B+C at item_bytes (issue #63)."""
    # 2 fp16 operands (2*2) + 3 fp32 temporaries (3*4) = 16
    assert _in_core_bytes_per_elem(Config(dtype="fp16", accumulate_fp32=True)) == 16
    # no upcast: A,B,C all fp16 -> 3*2
    assert _in_core_bytes_per_elem(Config(dtype="fp16", accumulate_fp32=False)) == 6
    # fp32: A,B,C -> 3*4
    assert _in_core_bytes_per_elem(Config(dtype="fp32")) == 12


def test_fits_in_core_rejects_fp16_upcast_boundary():
    """A budget that clears the naive 3*n^2*item estimate but not the true
    fp16+fp32 residency must now fall out of core instead of OOMing (issue #63)."""
    n = 8192
    cfg = Config(dtype="fp16", accumulate_fp32=True, vram_fraction=0.6)
    naive_need = 3 * n * n * cfg.item_bytes            # old under-budget: 6 n^2
    real_need = n * n * _in_core_bytes_per_elem(cfg)   # true peak: 16 n^2
    assert real_need > naive_need
    # Free VRAM sized to clear the naive estimate but not the true one.
    free = int(naive_need / cfg.vram_fraction) + 4096
    assert naive_need <= free * cfg.vram_fraction < real_need
    assert _fits_in_core(n, cfg, _FakeBackend(free)) is False


def test_fits_in_core_accepts_when_true_budget_fits():
    n = 4096
    cfg = Config(dtype="fp16", accumulate_fp32=True, vram_fraction=0.6)
    real_need = n * n * _in_core_bytes_per_elem(cfg)
    free = int(real_need / cfg.vram_fraction) + 4096   # clears the true need
    assert _fits_in_core(n, cfg, _FakeBackend(free)) is True


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
