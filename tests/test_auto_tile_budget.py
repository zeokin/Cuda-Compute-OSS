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
    # acc fp32 (4) + four fp32 operand-sized tiles (4*4) -- see
    # test_tile_workspace_counts_the_transient_fifth_tile for why it's four.
    assert _tile_workspace_bytes_per_elem(fp16_acc) == 20

    fp16_raw = Config(dtype="fp16", accumulate_fp32=False)
    # acc fp16 (2) + four fp16 operand-sized tiles (4*2)
    assert _tile_workspace_bytes_per_elem(fp16_raw) == 10


def test_tile_workspace_counts_the_bmm_output_tile():
    """Regression for #95: the per-step working set holds a fourth T×T tile —
    acc + A + B + the GEMM output `prod` — so the model must budget the output
    tile too, not just acc + 2 operands."""
    for cfg in (
        Config(dtype="fp32"),
        Config(dtype="fp16", accumulate_fp32=True),
        Config(dtype="fp16", accumulate_fp32=False),
    ):
        operand = _tile_operand_bytes(cfg)
        # Must be at least acc + three operand tiles (two operands + output) --
        # the current model charges a fourth on top; see the transient-fifth-
        # tile test below for why.
        assert _tile_workspace_bytes_per_elem(cfg) >= cfg.acc_dtype.itemsize + 3 * operand


def test_tile_workspace_counts_the_transient_fifth_tile():
    """_gemm_tiled_sync reassigns a_dev, b_dev, and prod every k-iteration
    (e.g. `a_dev = backend.to_device(...)`). Python evaluates the right-hand
    side -- allocating the NEW tile -- before the name rebinds and the OLD
    tile's reference drops, so whichever of those three lines is executing has
    the previous iteration's tile it's about to replace still live alongside
    the freshly allocated one: a fifth T×T tile, momentarily, on top of the
    steady-state four (acc, a_dev, b_dev, prod). Budgeting only four
    under-counts true peak device usage by one operand-sized tile."""
    for cfg in (
        Config(dtype="fp32"),
        Config(dtype="fp16", accumulate_fp32=True),
        Config(dtype="fp16", accumulate_fp32=False),
    ):
        operand = _tile_operand_bytes(cfg)
        assert _tile_workspace_bytes_per_elem(cfg) == cfg.acc_dtype.itemsize + 4 * operand

    # fp32 concretely: 4 (acc) + 4*4 (four operand-sized tiles) = 20, not 16.
    assert _tile_workspace_bytes_per_elem(Config(dtype="fp32")) == 20


def _legacy_auto_tile(n: int, cfg: Config, free_bytes: int) -> int:
    """Pre-fix estimator that sized fp16 operand tiles at item_bytes."""
    budget = int(free_bytes * cfg.vram_fraction)
    per_elem = cfg.acc_dtype.itemsize + 2 * cfg.item_bytes
    t = int(math.sqrt(max(1, budget) / per_elem))
    t = min(t, n)
    t = max(128, (t // 128) * 128)
    return min(t, n)


def test_config_rejects_non_positive_tile():
    """A non-positive tile makes gemm._tiles empty, silently returning an
    uninitialised C. Config must reject it at construction (issue #80)."""
    for bad in (0, -1, -5):
        try:
            Config(tile=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Config(tile={bad}) should raise ValueError")


def test_config_accepts_positive_tile_and_none():
    # Both valid tile specifications must still construct without error.
    assert Config(tile=None).tile is None
    assert Config(tile=256).tile == 256


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


def test_auto_tile_tiny_budget_stays_within_budget():
    """Tiny free-memory reports must not be rounded up to a 128 tile.

    The 128-alignment is a performance preference, not permission to exceed
    ``vram_fraction``. If the raw budget only allows a smaller tile, return it.
    """
    cfg = Config(dtype="fp32", vram_fraction=0.6)
    backend = _FakeBackend(16 * 1024)
    t = auto_tile(4096, cfg, backend)
    budget = int(backend.free_compute_bytes() * cfg.vram_fraction)
    need = t * t * _tile_workspace_bytes_per_elem(cfg)
    assert 1 <= t < 128
    assert need <= budget


def test_auto_tile_keeps_128_alignment_when_budget_allows():
    cfg = Config(dtype="fp32", vram_fraction=0.6)
    backend = _FakeBackend(64 * 1024**2)
    t = auto_tile(8192, cfg, backend)
    assert t >= 128
    assert t % 128 == 0


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
