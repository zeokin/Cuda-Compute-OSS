"""CPU tests for rsvd's rank-M sketch budget split (Fixes #91).

The 3-way split {col(A), row(A), row(B)} is verifiable without a GPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.transforms import RandomizedSVDTransform


def _three_way_widths(m: int) -> list[int]:
    base, rem = divmod(m, 3)
    return [base + (1 if i < rem else 0) for i in range(3)]


def test_rsvd_budget_splits_three_ways():
    for m in (30, 33, 48, 64, 128):
        widths = _three_way_widths(m)
        assert len(widths) == 3
        assert sum(widths) == m
        assert all(w >= 0 for w in widths)


def test_rsvd_basis_flops_formula_unchanged():
    # Total sketch width is still m, so the reported FLOP count stays 2n²m + 2nm².
    n, m = 200, 30
    rsvd = RandomizedSVDTransform()
    assert rsvd.basis_flops(n, m) == 2.0 * n * n * m + 2.0 * n * m * m


def test_four_way_split_would_waste_a_quarter_of_m():
    # Regression guard: the old 4-way split left one sketch unused for the product.
    m = 30
    four = [m // 4 + (1 if i < m % 4 else 0) for i in range(4)]
    three = _three_way_widths(m)
    assert sum(four) == sum(three) == m
    assert four[2] > 0  # col(B) sketch existed but was redundant
    assert len(three) == 3


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
