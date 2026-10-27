# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Delays that change over time: per-period EM and plug-in delay matrices."""

import numpy as np
import pytest

import epidemic_dates as ed
from epidemic_dates.methods import delay_periods

CUTOFF = 149


def simulated(seed, cutoff=CUTOFF):
    """Scenario F (mean delay 12 to 4 days), cut at `cutoff`.

    Returns the observed line list, the onsets of cases reported by the
    cut-off and the eventual onsets.
    """
    rng = np.random.default_rng(seed)
    onset, report, missing = ed.simulate("F", ed.scenario_setup("F"), rng)
    keep = report <= cutoff
    obs = np.where(missing[keep], np.nan, onset[keep]).astype(float)
    reported = np.bincount(onset[keep], minlength=cutoff + 1)[: cutoff + 1]
    eventual = np.bincount(onset[onset <= cutoff], minlength=cutoff + 1)[
        : cutoff + 1
    ]
    return obs, report[keep], reported, eventual


def weekly(x, cutoff=CUTOFF):
    """Sum into full weeks ending on the cut-off."""
    n = (cutoff + 1) // 7
    return np.asarray(x)[cutoff + 1 - 7 * n :].reshape(n, 7).sum(1)


def test_delay_periods_end_on_the_cutoff():
    assert delay_periods(9, None).tolist() == [0] * 10
    assert delay_periods(9, 4).tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert delay_periods(9, 10).tolist() == [0] * 10
    assert delay_periods(9, 1).tolist() == list(range(10))
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError, match="delay_period"):
            delay_periods(9, bad)


def test_per_period_em_beats_static_em_when_delays_fall():
    errs = {None: [], 14: []}
    now = {None: [], 14: []}
    for seed in (1, 2, 3):
        obs, report, reported, eventual = simulated(seed)
        for period in errs:
            res = ed.impute_linelist(
                obs, report, cutoff=CUTOFF, delay_period=period
            )
            err = np.abs(weekly(res["imputed"]["em"]) - weekly(reported))
            errs[period].append(err.sum() / reported.sum())
            nc = ed.nowcast_linelist(
                obs,
                report,
                cutoff=CUTOFF,
                delay_period=period,
                n_samples=1,
                bootstrap=False,
            )
            rel = weekly(nc["point"])[-4:] / weekly(eventual)[-4:] - 1
            now[period].append(np.mean(np.abs(rel)))
    assert np.mean(errs[14]) < np.mean(errs[None])
    assert np.mean(now[14]) < 0.75 * np.mean(now[None])


def test_per_period_delays_follow_the_trend():
    obs, report, _, _ = simulated(1)
    res = ed.impute_linelist(obs, report, cutoff=CUTOFF, delay_period=28)
    by_period = res["em_delay_by_period"]
    means = by_period @ np.arange(ed.DMAX + 1)
    assert by_period.shape == (6, ed.DMAX + 1)
    assert np.allclose(by_period.sum(1), 1)
    assert res["delay_period_start"].tolist() == [0, 10, 38, 66, 94, 122]
    assert means[1] > means[-1] + 3
    # the static delay sits between the first and last periods
    static = ed.impute_linelist(obs, report, cutoff=CUTOFF)["em_delay"]
    assert means[-1] < static @ np.arange(ed.DMAX + 1) < means[1]


def test_default_is_one_static_delay():
    obs, report, _, _ = simulated(2, cutoff=90)
    res = ed.impute_linelist(obs, report, cutoff=90)
    assert res["em_delay_by_period"].shape == (1, ed.DMAX + 1)
    assert np.array_equal(res["em_delay_by_period"][0], res["em_delay"])
    assert res["delay_period_start"].tolist() == [0]
    # one period covering every day is the same as no periods
    one = ed.impute_linelist(obs, report, cutoff=90, delay_period=91)
    assert np.array_equal(one["imputed"]["em"], res["imputed"]["em"])
    assert np.array_equal(one["em_delay"], res["em_delay"])
    nc = ed.nowcast_linelist(obs, report, cutoff=90, n_samples=20)
    nc1 = ed.nowcast_linelist(
        obs, report, cutoff=90, n_samples=20, delay_period=200
    )
    assert np.array_equal(nc["samples"], nc1["samples"], equal_nan=True)


def test_a_period_with_no_cases_takes_the_previous_delay():
    onset = np.r_[np.arange(14), np.arange(28, 42)].repeat(5)
    delays = np.r_[np.tile([0, 1, 2, 3, 5], 14), np.tile([2, 4, 6, 7, 9], 14)]
    report = onset + delays
    res = ed.impute_linelist(
        onset.astype(float), report, cutoff=55, delay_period=14
    )
    # periods start at days 0, 14, 28 and 42; the second and last have
    # no cases
    by_period = res["em_delay_by_period"]
    assert res["delay_period_start"].tolist() == [0, 14, 28, 42]
    assert np.allclose(by_period[1], by_period[0], atol=1e-9)
    assert np.allclose(by_period[3], by_period[2], atol=1e-9)
    assert not np.allclose(by_period[2], by_period[0], atol=1e-3)
    assert np.all(np.isfinite(res["imputed"]["em"]))


def test_short_line_list():
    onset = np.array([0, 2, np.nan, 5, np.nan])
    report = np.array([3, 6, 7, 9, 12])
    res = ed.impute_linelist(onset, report, cutoff=12, delay_period=4)
    assert res["em_delay_by_period"].shape == (4, ed.DMAX + 1)
    assert np.allclose(res["em_delay_by_period"].sum(1), 1)
    assert res["imputed"]["em"].sum() == pytest.approx(5)
    nc = ed.nowcast_linelist(
        onset, report, cutoff=12, delay_period=4, n_samples=50
    )
    assert nc["samples"].shape == (50, 13)
    assert nc["delay_by_period"].shape == (4, ed.DMAX + 1)
    days_left = np.minimum(12 - np.arange(13), ed.DMAX)
    period = delay_periods(12, 4)
    cdf = np.cumsum(nc["delay_by_period"], axis=1)
    assert np.allclose(nc["reported"], cdf[period, days_left])


def test_a_given_delay_is_held_fixed():
    onset = np.array([0, 2, np.nan, 5, np.nan])
    report = np.array([3, 6, 7, 9, 12])
    pmf = ed.gamma_pmf(4, 3)
    res = ed.impute_linelist(onset, report, delay=pmf)
    assert np.allclose(res["em_delay"], pmf)
    assert set(res["imputed"]) == {"backsampling", "mean", "em", "forward"}
    fwd = res["imputed"]["forward"] - res["complete"]
    assert fwd.sum() + res["imputed_before_day0"]["forward"] == (
        pytest.approx(2)
    )
    with pytest.raises(ValueError, match="not both"):
        ed.impute_linelist(onset, report, delay=pmf, delay_period=4)


def test_a_constant_delay_matrix_matches_a_static_delay():
    obs, report, _, _ = simulated(3, cutoff=80)
    pmf = ed.gamma_pmf(6, 5)
    matrix = np.tile(pmf, (81, 1))
    a = ed.impute_linelist(obs, report, cutoff=80, delay=pmf)
    b = ed.impute_linelist(obs, report, cutoff=80, delay=matrix)
    assert np.allclose(a["imputed"]["em"], b["imputed"]["em"])
    assert np.allclose(b["em_delay"], pmf)
    assert b["em_delay_by_period"].shape == (81, ed.DMAX + 1)
    assert b["delay_period_start"].tolist() == list(range(81))
    # forward renormalises over onset days 0..t with a matrix
    n_missing = b["missing_by_report"].sum()
    fwd = b["imputed"]["forward"] - b["complete"]
    assert fwd.sum() == pytest.approx(n_missing)
    with pytest.raises(ValueError, match="one row per onset day"):
        ed.impute_linelist(obs, report, cutoff=80, delay=matrix[:-1])


def test_the_true_delay_matrix_beats_a_static_plug_in():
    setup = ed.scenario_setup("F")
    err = {"true": [], "static": []}
    for seed in (1, 2):
        obs, report, _, eventual = simulated(seed)
        for name, delay in [
            ("true", setup["fmat"][: CUTOFF + 1]),
            ("static", ed.gamma_pmf(8, 8)),
        ]:
            nc = ed.nowcast_linelist(
                obs,
                report,
                cutoff=CUTOFF,
                delay=delay,
                n_samples=1,
                bootstrap=False,
            )
            rel = weekly(nc["point"])[-4:] / weekly(eventual)[-4:] - 1
            err[name].append(np.mean(np.abs(rel)))
    assert np.mean(err["true"]) < 0.5 * np.mean(err["static"])


@pytest.mark.parametrize("method", ed.NOWCAST_METHODS)
def test_nowcast_methods_run_with_periods(method):
    obs, report, _, _ = simulated(4, cutoff=70)
    res = ed.nowcast_linelist(
        obs,
        report,
        cutoff=70,
        method=method,
        delay_period=21,
        n_samples=20,
    )
    ok = np.isfinite(res["point"])
    assert ok[:60].all()
    assert np.all(np.isfinite(res["samples"][:, ok]))
    assert res["period_start"].tolist() == [0, 8, 29, 50]
