"""``--spectral-alpha`` exposes storage's k^-alpha decay exponent -- previously
the only ``storage.generate`` knob with no CLI path, so ``--fill
decaying-spectrum`` was stuck at the default alpha=1.0. These pure-NumPy checks
pin (a) that the flag is parsed and (b) that the exponent actually steepens the
singular-value decay, so a larger alpha concentrates more energy in the leading
components.
"""
from __future__ import annotations

import numpy as np

from strategy import storage
from strategy.cli import build_parser


def test_spectral_alpha_is_parsed():
    args = build_parser().parse_args(["--fill", "decaying-spectrum",
                                      "--spectral-alpha", "2.5"])
    assert args.spectral_alpha == 2.5
    # default is 1.0 (matches storage.generate's default)
    assert build_parser().parse_args([]).spectral_alpha == 1.0


def _leading_energy_fraction(mat, k):
    s = np.linalg.svd(np.asarray(mat, dtype=np.float64), compute_uv=False)
    return float((s[:k] ** 2).sum() / (s ** 2).sum())


def test_larger_alpha_steepens_the_spectrum():
    n, R = 96, 32
    gentle = storage.generate(n, np.float64, False, None, seed=0,
                              fill="decaying-spectrum", data_rank=R,
                              spectral_alpha=0.5)
    steep = storage.generate(n, np.float64, False, None, seed=0,
                             fill="decaying-spectrum", data_rank=R,
                             spectral_alpha=3.0)
    # A steeper exponent must pack more of the energy into the top few components.
    assert _leading_energy_fraction(steep, 4) > _leading_energy_fraction(gentle, 4)


def test_alpha_zero_is_flat_like_lowrank_scale():
    # alpha=0 -> k^0 = 1 for every component -> uniform weights (no decay).
    n, R = 64, 16
    flat = storage.generate(n, np.float64, False, None, seed=1,
                            fill="decaying-spectrum", data_rank=R,
                            spectral_alpha=0.0)
    s = np.linalg.svd(np.asarray(flat, dtype=np.float64), compute_uv=False)
    # rank is exactly R and the top-R singular values are of comparable scale
    # (no k^-alpha taper), unlike a steep spectrum.
    assert np.count_nonzero(s > 1e-9 * s[0]) == R
    assert s[R - 1] / s[0] > 0.05


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
