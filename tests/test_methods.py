# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
import numpy as np
import pytest

from epidemic_dates.delays import DMAX
from epidemic_dates.methods import (
    METHODS,
    em_backproject,
    em_fit,
    hub_allocate,
    mean_shift_allocate,
    round_half_up,
)


def test_methods_are_an_immutable_tuple():
    assert METHODS == ("hub", "mean", "forward", "em")
    assert isinstance(METHODS, tuple)


@pytest.mark.parametrize(
    ("x", "expected"),
    [(0.5, 1), (1.5, 2), (2.5, 3), (2.4999, 2), (-0.5, 0), (7.0, 7)],
)
def test_round_half_up(x, expected):
    assert round_half_up(x) == expected


def test_round_half_up_rejects_nan():
    with pytest.raises(ValueError, match="non-finite"):
        round_half_up(float("nan"))


def test_hub_allocate_spreads_by_delay():
    missing = np.array([0.0, 0.0, 0.0, 4.0])
    pmf = np.array([0.25, 0.25, 0.5])
    assert np.allclose(hub_allocate(missing, pmf), [0, 2, 1, 1])


def test_hub_allocate_drops_mass_before_day_zero():
    missing = np.array([0.0, 4.0])
    pmf = np.array([0.25, 0.25, 0.5])
    alloc = hub_allocate(missing, pmf)
    assert np.allclose(alloc, [1, 1])
    assert np.isclose(missing.sum() - alloc.sum(), 2)


def test_hub_allocate_handles_series_shorter_than_pmf():
    # Regression: a line list shorter than the maximum delay used to fail
    # with a broadcasting error.
    alloc = hub_allocate(np.array([1.0, 2.0]), np.full(DMAX + 1, 1 / 61))
    assert alloc.shape == (2,)


def test_mean_shift_allocate():
    missing = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(mean_shift_allocate(missing, 0), missing)
    assert np.allclose(mean_shift_allocate(missing, 2), [3, 4, 0, 0])
    assert np.allclose(mean_shift_allocate(missing, 10), 0)


def test_mean_shift_allocate_rejects_negative_shift():
    with pytest.raises(ValueError, match="non-negative"):
        mean_shift_allocate(np.ones(3), -1)


def _complete_and_missing(seed=4, cutoff=80, n=600):
    rng = np.random.default_rng(seed)
    onset = rng.integers(0, cutoff, size=n)
    delay = rng.integers(0, 15, size=n)
    report = onset + delay
    keep = report <= cutoff
    onset, delay, report = onset[keep], delay[keep], report[keep]
    missing = rng.random(onset.size) < 0.3
    return (onset[~missing], delay[~missing]), report[missing], cutoff


def test_em_fit_allocates_every_missing_case():
    complete, missing_report, cutoff = _complete_and_missing()
    fit = em_fit(complete, missing_report, cutoff, n_iter=30)
    assert fit.allocation.shape == (cutoff + 1,)
    assert fit.onsets.shape == (cutoff + 1,)
    assert fit.delay.shape == (DMAX + 1,)
    assert np.isclose(fit.allocation.sum(), missing_report.size)
    assert np.isclose(fit.delay.sum(), 1)
    assert np.all(fit.allocation >= 0)


def test_em_fit_recovers_delay_support():
    complete, missing_report, cutoff = _complete_and_missing()
    fit = em_fit(complete, missing_report, cutoff, n_iter=50)
    assert fit.delay[15:].sum() < 1e-3


def test_em_backproject_is_em_fit_allocation():
    complete, missing_report, cutoff = _complete_and_missing()
    assert np.array_equal(
        em_backproject(complete, missing_report, cutoff, n_iter=10),
        em_fit(complete, missing_report, cutoff, n_iter=10).allocation,
    )


def test_em_fit_with_no_missing_cases():
    complete, _, cutoff = _complete_and_missing()
    fit = em_fit(complete, np.array([], dtype=int), cutoff, n_iter=5)
    assert np.allclose(fit.allocation, 0)


def test_em_fit_with_no_cases_is_finite():
    empty = np.array([], dtype=int)
    fit = em_fit((empty, empty), empty, 10, n_iter=2)
    assert np.all(np.isfinite(fit.allocation))


@pytest.mark.parametrize(
    ("complete", "missing_report", "cutoff", "n_iter", "match"),
    [
        (([0], [0]), [0], -1, 5, "cutoff"),
        (([0], [0]), [0], 5, 0, "n_iter"),
        (([0, 1], [0]), [0], 5, 5, "same length"),
        (([-1], [0]), [0], 5, 5, "onset days"),
        (([0], [DMAX + 1]), [0], 100, 5, "delays"),
        (([4], [3]), [0], 5, 5, "reported by the cutoff"),
        (([0], [0]), [6], 5, 5, "missing_report"),
        (([0.5], [0]), [0], 5, 5, "whole numbers"),
        (([0], [0]), [np.nan], 5, 5, "NaN"),
        (([[0]], [0]), [0], 5, 5, "one-dimensional"),
    ],
)
def test_em_fit_rejects_bad_inputs(
    complete, missing_report, cutoff, n_iter, match
):
    with pytest.raises(ValueError, match=match):
        em_fit(complete, missing_report, cutoff, n_iter=n_iter)
