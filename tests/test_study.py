# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
import csv

import numpy as np
import pytest

from onset_imputation.delays import DMAX
from onset_imputation.methods import METHODS
from onset_imputation.scenarios import SCEN, scenario_setup
from onset_imputation.study import (
    OFF,
    analytic,
    first_negative_week,
    growth_rate,
    run_rep,
    run_scenario,
    run_study,
    summarise,
    weekly,
    write_results,
)


def test_weekly_bins_days_into_weeks():
    out = weekly([0, 6, 7, 13, 14], n_weeks=3)
    assert np.array_equal(out, [2, 2, 1])


def test_weekly_drops_days_outside_range():
    out = weekly([-OFF, -1, 0, 7 * 3], n_weeks=3)
    assert np.array_equal(out, [1, 0, 0])


def test_weekly_with_weights():
    out = weekly(np.arange(14), weights=np.full(14, 0.5), n_weeks=2)
    assert np.allclose(out, [3.5, 3.5])


def test_growth_rate_matches_log_slope():
    counts = np.exp(0.1 * 7 * np.arange(6)) - 0.5
    rate = growth_rate(counts)
    assert np.isnan(rate[0]) and np.isnan(rate[-1])
    assert np.allclose(rate[1:-1], 0.1)


def test_first_negative_week():
    rates = np.array(
        [
            [1, 1, 1, 1, 1, -1, -1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    out = first_negative_week(rates)
    assert out[0] == 5
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_run_rep_returns_each_method_at_each_cutoff():
    setup = scenario_setup("A")
    rep = run_rep("A", setup, np.random.default_rng(1), [70, 179], None)
    assert set(rep) == {70, 179}
    for res in rep.values():
        assert set(METHODS) | {"truth", "n"} <= set(res)
    # Retrospectively every case is reported and EM keeps every case.
    assert np.isclose(rep[179]["em"].sum(), rep[179]["truth"].sum())


def test_run_rep_uses_time_varying_forward_pmf():
    setup = scenario_setup("F")
    rep = run_rep("F", setup, np.random.default_rng(1), [120], setup["fmat"])
    assert rep[120]["forward"].sum() > 0


def test_run_scenario_returns_each_method():
    rows = run_scenario("A", reps=2, cutoffs=[179])
    assert {r["method"] for r in rows} == set(METHODS)
    assert all(np.isfinite(r["relbias_mean"]) for r in rows)
    assert all(r["retro"] == 1 for r in rows)
    assert all("peak_shift_wk" in r for r in rows)


def test_run_scenario_default_cutoffs_and_columns():
    rows = run_scenario("W", reps=2)
    retro = SCEN["W"].days - 1 + DMAX
    assert [r["cutoff"] for r in rows[:: len(METHODS)]] == [retro, 90, 120]
    realtime = [r for r in rows if not r["retro"]]
    assert all("relbias_last2" in r for r in realtime)
    assert all("relbias_cp" in r for r in rows if r["cutoff"] == retro)
    assert all("zero_cross_lag_wk" in r for r in rows if r["retro"])


def test_run_scenario_is_reproducible():
    a = run_scenario("B0", reps=2, seed=3, cutoffs=[90])
    b = run_scenario("B0", reps=2, seed=3, cutoffs=[90])
    assert a == b


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "Z"}, "unknown scenario"),
        ({"name": "A", "reps": 0}, "reps"),
        ({"name": "A", "cutoffs": [-1]}, "non-negative"),
    ],
)
def test_run_scenario_rejects_bad_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        run_scenario(**kwargs)


def test_summarise_without_change_point_omits_cp_columns():
    setup = scenario_setup("A")
    rng = np.random.default_rng(2)
    reps = [run_rep("A", setup, rng, [90], None) for _ in range(2)]
    rows = summarise("A", setup, 90, reps, retro=False)
    assert len(rows) == len(METHODS)
    assert all("relbias_cp" not in r for r in rows)


def test_run_study_seed_does_not_depend_on_subset():
    # Regression: the i-th scenario run used seed + i, so `--scenarios W`
    # gave different numbers from the full run.
    alone = run_study(reps=1, scenarios=["W"], verbose=False)
    with_other = run_study(reps=1, scenarios=["B0", "W"], verbose=False)
    w_rows = [r for r in with_other if r["scenario"] == "W"]
    assert alone == w_rows
    seed = 20261027 + list(SCEN).index("W")
    assert alone == run_scenario("W", reps=1, seed=seed)


def test_run_study_prints_progress(capsys):
    run_study(reps=1, scenarios=["B0"], verbose=True)
    assert "done B0" in capsys.readouterr().out


def test_run_study_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="unknown scenario"):
        run_study(reps=1, scenarios=["Z"], verbose=False)


def test_analytic_returns_lines():
    lines = analytic()
    assert len(lines) == 6
    assert lines[0].startswith("r=+0.050")


def test_write_results(tmp_path):
    rows = [
        {"scenario": "A", "value": 0.123456, "n": 3},
        {"scenario": "B", "extra": np.float64(1.0)},
    ]
    out = tmp_path / "nested" / "dir"
    write_results(rows, out)
    with (out / "results.csv").open() as fh:
        read = list(csv.DictReader(fh))
    assert list(read[0]) == ["scenario", "value", "n", "extra"]
    assert read[0]["value"] == "0.1235"
    assert read[1]["extra"] == "1.0000"
    assert (out / "analytic.txt").read_text().count("\n") == 6
