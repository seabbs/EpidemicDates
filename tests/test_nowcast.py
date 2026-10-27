# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
import csv

import numpy as np
import pytest

import epidemic_dates as ed

CUTOFF = 110


def simulated(seed, name="C", cutoff=CUTOFF):
    """A simulated line list cut at `cutoff`, with the eventual truth."""
    rng = np.random.default_rng(seed)
    onset, report, missing = ed.simulate(name, ed.scenario_setup(name), rng)
    truth = np.bincount(
        onset[onset <= cutoff].astype(int), minlength=cutoff + 1
    )[: cutoff + 1]
    keep = report <= cutoff
    obs = np.where(missing[keep], np.nan, onset[keep])
    return obs, report[keep], truth


@pytest.fixture(scope="module")
def small():
    obs, report, _ = simulated(1, "Csmall", 100)
    return obs, report


def coverage(res, truth, days=28):
    """Coverage of the 50% and 95% intervals over the last `days` days."""
    s = res["samples"][:, -days:]
    ok = np.all(np.isfinite(s), 0)
    q = np.quantile(s[:, ok], [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    t = truth[-days:][ok]
    c50 = np.mean((t >= q[1]) & (t <= q[3]))
    c95 = np.mean((t >= q[0]) & (t <= q[4]))
    return c50, c95, q[2], t


@pytest.mark.parametrize(
    "method,delay,bootstrap",
    [
        ("em", "true", False),
        ("backsampling", "true", False),
        ("em", None, True),
    ],
)
def test_nowcast_covers_the_eventual_truth(method, delay, bootstrap):
    c50, c95, rel = [], [], []
    for seed in (1, 2, 3):
        obs, report, truth = simulated(seed)
        res = ed.nowcast_linelist(
            obs,
            report,
            cutoff=CUTOFF,
            method=method,
            delay=ed.F_DEFAULT if delay else None,
            n_samples=300,
            seed=seed,
            bootstrap=bootstrap,
        )
        a, b, med, t = coverage(res, truth)
        c50.append(a)
        c95.append(b)
        rel.append(med.sum() / t.sum() - 1)
    assert 0.3 <= np.mean(c50) <= 0.85
    assert np.mean(c95) >= 0.85
    assert abs(np.mean(rel)) < 0.08


def test_median_is_close_to_the_truth_by_week():
    obs, report, truth = simulated(4)
    res = ed.nowcast_linelist(
        obs, report, cutoff=CUTOFF, delay=ed.F_DEFAULT, n_samples=300
    )
    n_weeks = res["week_start"].size
    tw = truth[CUTOFF + 1 - 7 * n_weeks :].reshape(n_weeks, 7).sum(1)
    med = np.median(res["weekly_samples"], 0)
    for w in range(n_weeks - 3, n_weeks):
        assert abs(med[w] / tw[w] - 1) < 0.15, w


def test_quantiles_are_monotone_and_ordered(small):
    res = ed.nowcast_linelist(*small, n_samples=200)
    for table, key in [
        (res["daily_quantiles"], "target_end_date"),
        (res["weekly_quantiles"], "target_end_date"),
    ]:
        by_date = {}
        for row in table:
            by_date.setdefault(row[key], []).append(
                (row["quantile"], row["value"])
            )
        assert by_date
        for rows in by_date.values():
            levels = [q for q, _ in rows]
            values = [v for _, v in rows]
            assert levels == list(ed.QUANTILES)
            assert np.all(np.diff(values) >= 0)


def test_reproducible_with_a_seed(small):
    a = ed.nowcast_linelist(*small, n_samples=50, seed=7)
    b = ed.nowcast_linelist(*small, n_samples=50, seed=7)
    c = ed.nowcast_linelist(*small, n_samples=50, seed=8)
    assert np.array_equal(a["samples"], b["samples"], equal_nan=True)
    assert a["weekly_quantiles"] == b["weekly_quantiles"]
    assert not np.array_equal(a["samples"], c["samples"], equal_nan=True)


@pytest.mark.parametrize("method", ed.NOWCAST_METHODS)
def test_every_method_matches_the_truncation_corrected_imputation(
    small, method
):
    obs, report = small
    res = ed.nowcast_linelist(
        obs, report, method=method, delay=ed.F_DEFAULT, n_samples=20
    )
    imp = ed.impute_linelist(obs, report)
    key = {"forward": "em"}.get(method, method)
    reported = np.cumsum(ed.F_DEFAULT)[
        np.minimum(res["cutoff"] - np.arange(res["cutoff"] + 1), ed.DMAX)
    ]
    assert np.allclose(res["reported"], reported)
    ok = reported >= 0.05
    assert np.all(np.isnan(res["point"][~ok]))
    if method in ("backsampling", "mean"):
        expected = imp["imputed"][key] / reported
        assert np.allclose(res["point"][ok], expected[ok])
    samples = res["samples"]
    assert np.all(np.isfinite(samples[:, ok]))
    assert np.all(np.isnan(samples[:, ~ok]))
    # samples are whole numbers and at least the cases already reported
    assert np.all(samples[:, ok] == np.round(samples[:, ok]))
    assert np.all(samples[:, ok] >= imp["complete"][ok])


def test_em_with_a_given_delay_matches_em_fit(small):
    obs, report = small
    res = ed.nowcast_linelist(obs, report, delay=ed.F_DEFAULT, n_samples=5)
    miss = np.isnan(obs)
    fit = ed.em_fit(
        (obs[~miss].astype(int), (report[~miss] - obs[~miss]).astype(int)),
        report[miss].astype(int),
        res["cutoff"],
        delay=ed.F_DEFAULT,
    )
    assert np.allclose(fit.delay, ed.F_DEFAULT)
    ok = np.isfinite(res["point"])
    assert np.allclose(res["point"][ok], fit.onsets[ok], rtol=1e-6)


def test_days_under_min_reported_have_no_estimate(small):
    res = ed.nowcast_linelist(*small, n_samples=20, min_reported=0.3)
    none = res["reported"] < 0.3
    assert none.any()
    assert np.all(np.isnan(res["point"][none]))
    days = {row["target_end_date"] for row in res["daily_quantiles"]}
    assert days == set(np.flatnonzero(~none).tolist())
    # weeks with any such day have no estimate either
    assert np.isnan(res["point_weekly"][-1])
    last = res["week_start"][-1] + 6
    assert all(r["target_end_date"] != last for r in res["weekly_quantiles"])


def test_weeks_end_on_the_cutoff_and_dates_follow_origin(small):
    res = ed.nowcast_linelist(
        *small, n_samples=20, origin="2026-05-08", min_reported=0.01
    )
    cutoff = res["cutoff"]
    assert res["week_start"][-1] + 6 == cutoff
    assert np.all(np.diff(res["week_start"]) == 7)
    assert res["week_start"][0] >= 0
    row = res["weekly_quantiles"][-1]
    assert row["target_end_date"] == "2026-08-16"  # day 100
    assert row["week_start"] == "2026-08-10"
    assert res["daily_quantiles"][0]["target_end_date"] == "2026-05-08"
    n_weeks = res["week_start"].size
    tail = res["samples"][:, cutoff + 1 - 7 * n_weeks :]
    assert np.allclose(
        res["weekly_samples"],
        tail.reshape(len(tail), n_weeks, 7).sum(-1),
        equal_nan=True,
    )


def test_write_nowcast_csv(small, tmp_path):
    res = ed.nowcast_linelist(*small, n_samples=20)
    path = tmp_path / "weekly.csv"
    ed.write_nowcast_csv(res, path)
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == [
        "week_start",
        "target_end_date",
        "quantile",
        "value",
    ]
    assert len(rows) == len(res["weekly_quantiles"])
    ed.write_nowcast_csv(res, tmp_path / "daily.csv", by="day")
    header = (tmp_path / "daily.csv").read_text().splitlines()[0]
    assert header == "target_end_date,quantile,value"
    with pytest.raises(ValueError, match="by must be"):
        ed.write_nowcast_csv(res, path, by="month")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"method": "unknown"}, "method must be one of"),
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": 2.5}, "n_samples"),
        ({"n_iter": 0}, "n_iter"),
        ({"min_reported": 0}, "min_reported"),
        ({"min_reported": 1.5}, "min_reported"),
        ({"quantiles": []}, "quantiles"),
        ({"quantiles": [0.5, 1.2]}, "quantiles"),
        ({"delay": [0.0, 0.0]}, "positive mass"),
        ({"delay": [0.5, -0.1]}, "non-negative"),
        ({"delay": [[0.5, 0.5]]}, "one row per onset day"),
        ({"delay": [[[0.5, 0.5]]]}, "matrix of pmfs"),
        ({"delay": [0.5, 0.5], "delay_period": 7}, "not both"),
        ({"delay_period": 0}, "delay_period"),
        ({"delay_period": 2.5}, "delay_period"),
        ({"delay_shrinkage": 0}, "delay_shrinkage"),
        ({"cutoff": -1}, "cutoff"),
    ],
)
def test_input_validation(kwargs, match):
    onset = np.array([0, 1, np.nan, 3])
    report = np.array([2, 3, 4, 5])
    with pytest.raises(ValueError, match=match):
        ed.nowcast_linelist(onset, report, **kwargs)


def test_line_list_validation_matches_impute_linelist():
    with pytest.raises(ValueError, match="same length"):
        ed.nowcast_linelist([1, 2], [3])
    with pytest.raises(ValueError, match="whole numbers"):
        ed.nowcast_linelist([1.5], [3])
    with pytest.raises(ValueError, match="no complete cases"):
        ed.nowcast_linelist([np.nan, np.nan], [1, 2])
    res = ed.nowcast_linelist([0, 5, np.nan, 1], [2, 4, 3, 3], n_samples=5)
    assert res["dropped"] == 1  # onset after report
