"""CPU-only tests: storage.generate must reject a non-finite spectral_alpha at
its public boundary (not just the CLI).

A sign-only guard let NaN/Inf reach _fill_decaying_spectrum, whose k**-alpha
weights then went all-NaN (NaN alpha -> all-NaN A/B) or collapsed to [1, 0, ...]
(Inf alpha -> a silent rank-1 "decaying-spectrum" matrix). Any caller of the
runner/storage API -- not only the CLI -- must be protected.

Run:  python -m pytest tests/test_decaying_spectrum_alpha.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from strategy import storage


def _gen(alpha):
    return storage.generate(
        64, np.float64, False, None, seed=0, fill="decaying-spectrum",
        data_rank=16, spectral_alpha=alpha,
    )


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), float("-inf"), -0.5])
def test_generate_rejects_non_finite_or_negative_alpha(alpha):
    with pytest.raises(ValueError, match="spectral_alpha must be a finite number"):
        _gen(alpha)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.5])
def test_generate_accepts_finite_alpha_and_stays_finite(alpha):
    mat = _gen(alpha)
    assert np.isfinite(mat).all()


def test_non_finite_alpha_ignored_for_non_decaying_fills():
    # spectral_alpha is only consumed by the decaying-spectrum fill, so a stray
    # non-finite value must not break unrelated fills.
    mat = storage.generate(32, np.float64, False, None, seed=0, fill="random",
                           spectral_alpha=float("nan"))
    assert np.isfinite(mat).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
