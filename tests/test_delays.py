# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
import numpy as np
import pytest

from epidemic_dates.delays import DMAX, F_DEFAULT, gamma_pmf


def test_gamma_pmf_sums_to_one():
    assert abs(gamma_pmf(8, 8).sum() - 1) < 1e-12


def test_gamma_pmf_mean_matches_gamma_mean():
    d = np.arange(DMAX + 1)
    assert abs((d * gamma_pmf(8, 4)).sum() - 8) < 0.01


def test_gamma_pmf_is_non_negative_with_dmax_length():
    pmf = gamma_pmf(5, 3, dmax=20)
    assert pmf.shape == (21,)
    assert np.all(pmf >= 0)


def test_gamma_pmf_dmax_zero_is_a_point_mass():
    assert np.allclose(gamma_pmf(5, 3, dmax=0), [1.0])


def test_backward_mean_shorter_under_growth():
    f = gamma_pmf(8, 8)
    d = np.arange(f.size)
    b = f * np.exp(-0.05 * d)
    assert (d * b).sum() / b.sum() < (d * f).sum() - 1.5


@pytest.mark.parametrize(
    ("mean", "sd", "dmax", "match"),
    [
        (0, 1, 60, "mean"),
        (-1, 1, 60, "mean"),
        (np.nan, 1, 60, "mean"),
        (5, 0, 60, "sd"),
        (5, np.inf, 60, "sd"),
        (5, 1, -1, "dmax"),
    ],
)
def test_gamma_pmf_rejects_bad_inputs(mean, sd, dmax, match):
    with pytest.raises(ValueError, match=match):
        gamma_pmf(mean, sd, dmax=dmax)


def test_default_delay_is_read_only():
    assert np.allclose(F_DEFAULT, gamma_pmf(8, 8))
    with pytest.raises(ValueError, match="read-only"):
        F_DEFAULT[0] = 1.0
