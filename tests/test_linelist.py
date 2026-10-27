# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
import numpy as np
import pytest

import epidemic_dates as ed
from epidemic_dates.delays import DMAX
from epidemic_dates.linelist import impute_linelist
from epidemic_dates.methods import em_fit


@pytest.fixture(scope="module")
def scenario_c():
    """A line list from scenario C, cut at day 140, with the truth."""
    rng = np.random.default_rng(1)
    setup = ed.scenario_setup("C")
    onset, report, missing = ed.simulate("C", setup, rng)
    keep = report <= 140
    onset, report, missing = onset[keep], report[keep], missing[keep]
    observed = np.where(missing, np.nan, onset.astype(float))
    return onset, report, observed


def test_em_recovers_onsets(scenario_c):
    onset, report, observed = scenario_c
    cutoff = 140
    res = impute_linelist(observed, report, cutoff=cutoff)
    truth = np.bincount(onset, minlength=cutoff + 1)
    week = slice(56, 63)  # a busy week before the turn
    rel = res["imputed"]["em"][week].sum() / truth[week].sum() - 1
    assert abs(rel) < 0.1
    assert abs(res["imputed"]["em"].sum() - onset.size) < 1e-6 * onset.size


def test_outputs_have_documented_shapes(scenario_c):
    _, report, observed = scenario_c
    res = impute_linelist(observed, report, cutoff=140)
    for key in ("complete", "missing_by_report"):
        assert res[key].shape == (141,)
    for key in ("empirical_delay", "em_delay"):
        assert res[key].shape == (DMAX + 1,)
        assert np.isclose(res[key].sum(), 1)
    assert set(res["imputed"]) == {"hub", "mean", "em"}
    assert res["dropped"] == 0
    assert np.isclose(res["missing_by_report"].sum(), np.isnan(observed).sum())


def test_cutoff_defaults_to_last_report():
    res = impute_linelist([0, 1, np.nan], [2, 4, 5])
    assert res["complete"].shape == (6,)


def test_accepts_lists_with_none():
    res = impute_linelist([0, None, 2], [1, 3, 4], cutoff=4)
    assert res["missing_by_report"][3] == 1


def test_dropped_counts_out_of_range_cases():
    onset = [0, 5, 1, np.nan, -1, 0]
    report = [2, 3, 10, -1, 1, 200]
    # onset after report, after cutoff, report < 0, onset < 0, delay > DMAX
    res = impute_linelist(onset, report, cutoff=8)
    assert res["dropped"] == 5
    res = impute_linelist(onset, report, cutoff=200)
    assert res["dropped"] == 4


def test_short_linelist_runs():
    # Regression: a cut-off below DMAX made hub and mean fail with a
    # broadcasting error. This is the README example.
    onset = np.array([0, 2, np.nan, 5, np.nan])
    report = np.array([3, 6, 7, 9, 12])
    res = impute_linelist(onset, report, cutoff=12)
    for method in ("hub", "mean", "em"):
        assert res["imputed"][method].shape == (13,)
    assert np.isclose(res["imputed"]["em"].sum(), 5)


def test_imputed_before_day0_is_reported():
    # Regression: hub and mean discarded imputed onsets before day 0
    # without reporting them.
    onset = [0, 0, np.nan, np.nan]
    report = [2, 4, 1, 10]
    res = impute_linelist(onset, report, cutoff=10)
    n_missing = 2
    for method in ("hub", "mean"):
        total = res["imputed"][method].sum() - res["complete"].sum()
        lost = res["imputed_before_day0"][method]
        assert np.isclose(total + lost, n_missing)
    # delays 2 and 4: mean 3, so the case reported on day 1 is lost
    assert np.isclose(res["imputed_before_day0"]["mean"], 1)
    # hub: day-1 case lost with P(delay > 1) = 1
    assert np.isclose(res["imputed_before_day0"]["hub"], 1)


def test_mean_shift_rounds_halves_up():
    # Regression: Python's round() rounds 2.5 to 2; JavaScript rounds up.
    onset = [0, 0, np.nan]
    report = [2, 3, 10]
    res = impute_linelist(onset, report, cutoff=10)
    imputed = res["imputed"]["mean"] - res["complete"]
    assert np.isclose(imputed[7], 1)


def test_em_delay_is_returned():
    # Regression: the EM delay estimate was not returned.
    rng = np.random.default_rng(5)
    onset = rng.integers(0, 60, size=400)
    report = onset + rng.integers(0, 10, size=400)
    observed = np.where(rng.random(400) < 0.3, np.nan, onset)
    res = impute_linelist(observed, report, cutoff=70, n_iter=20)
    keep = report <= 70
    o, r = observed[keep], report[keep]
    m = np.isnan(o)
    fit = em_fit(
        (o[~m].astype(int), (r - o)[~m].astype(int)), r[m], 70, n_iter=20
    )
    assert np.allclose(res["em_delay"], fit.delay)
    assert res["em_delay"][12:].sum() < 1e-3


def test_no_complete_cases_is_a_clear_error():
    # Regression: this gave a NaN empirical delay.
    with pytest.raises(ValueError, match="no complete cases"):
        impute_linelist([np.nan, np.nan], [1, 2])
    with pytest.raises(ValueError, match="no complete cases"):
        impute_linelist([5, np.nan], [1, 2])


def test_nan_report_day_is_a_clear_error():
    # Regression: this failed inside numpy with an unclear message.
    with pytest.raises(ValueError, match="report day is missing"):
        impute_linelist([0, 1], [2, np.nan])
    with pytest.raises(ValueError, match="report day is missing"):
        impute_linelist([0, 1], [2, None])


@pytest.mark.parametrize(
    ("onset", "report", "cutoff", "match"),
    [
        ([], [], None, "empty"),
        ([0, 1], [1], None, "same length"),
        ([[0]], [[1]], None, "one-dimensional"),
        ([0.5, 1], [1, 2], None, "onset days must be whole"),
        ([0, 1], [1.5, 2], None, "report days must be whole"),
        ([np.inf, 1], [1, 2], None, "infinite"),
        ([0, 1], [np.inf, 2], None, "infinite"),
        ([0, 1], [1, 2], -1, "non-negative"),
        ([0, 1], [1, 2], 1.5, "whole number"),
    ],
)
def test_rejects_bad_inputs(onset, report, cutoff, match):
    with pytest.raises(ValueError, match=match):
        impute_linelist(onset, report, cutoff=cutoff)


def test_rejects_bad_n_iter():
    with pytest.raises(ValueError, match="n_iter"):
        impute_linelist([0, 1], [1, 2], n_iter=0)
