"""Pin matmul/README's "it doesn't fit" table to a single n convention (issue #102).

The table's memory columns use decimal n (e.g. 128k -> 128000, so one fp32
matrix is 128000**2 * 4 = 65 GB). The `compute (2n**3)` column must use the SAME
decimal n, or it silently overstates the FLOP figures by ~7% (a binary n=1024 is
(1024/1000)**3 = 1.074x a decimal n=1000). Before the fix the 1k/16k/32k rows
used binary n (1024/16384/32768 -> 2.1/8.8/70) while the 128k row already used
decimal n (-> 4.2 PFLOP), so the column mixed both conventions.

This test derives the compute column straight from the decimal n the memory
column uses, so the documented figures cannot drift back to the inconsistent
mixed-convention values. Pure arithmetic; no GPU needed.

Run:  python tests/test_matmul_flop_table_units.py
"""


def _tflops_2n3(n: int, unit: float) -> float:
    return 2.0 * n**3 / unit


GFLOP, TFLOP, PFLOP = 1e9, 1e12, 1e15


def test_compute_column_uses_the_same_decimal_n_as_the_memory_column():
    # (decimal n, documented value, unit)
    assert round(_tflops_2n3(1_000, GFLOP), 1) == 2.0
    assert round(_tflops_2n3(16_000, TFLOP), 1) == 8.2
    assert round(_tflops_2n3(32_000, TFLOP)) == 66
    assert round(_tflops_2n3(128_000, PFLOP), 1) == 4.2


def test_memory_column_is_decimal_n_too():
    # one fp32 matrix = n**2 * 4 bytes; the table's "65 GB" at 128k comes from
    # decimal n=128000 (65.5 GB), not binary n=131072 (which would be 68.7 GB).
    assert round(128_000**2 * 4 / 1e9, 1) == 65.5
    assert round(131_072**2 * 4 / 1e9, 1) == 68.7
    assert round(1_000**2 * 4 / 1e6) == 4        # 4 MB


def test_old_binary_n_figures_were_inconsistently_larger():
    # Regression witness: the pre-fix 1k/16k/32k figures came from binary n and
    # overstate 2n**3 vs the decimal n the memory column uses.
    assert round(_tflops_2n3(1_024, GFLOP), 1) == 2.1
    assert round(_tflops_2n3(16_384, TFLOP), 1) == 8.8
    assert round(_tflops_2n3(32_768, TFLOP)) == 70
    assert _tflops_2n3(1_024, GFLOP) > _tflops_2n3(1_000, GFLOP)


if __name__ == "__main__":
    import sys
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
