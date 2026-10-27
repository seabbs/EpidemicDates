# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
import numpy as np

import onset_imputation as sim
from onset_imputation.simulation import DMAX


def test_gamma_pmf_sums_to_one():
    assert abs(sim.gamma_pmf(8, 8).sum() - 1) < 1e-12


def test_gamma_pmf_mean_matches_gamma_mean():
    d = np.arange(DMAX + 1)
    assert abs((d * sim.gamma_pmf(8, 4)).sum() - 8) < 0.01


def test_backward_mean_shorter_under_growth():
    f = sim.gamma_pmf(8, 8)
    d = np.arange(f.size)
    b = f * np.exp(-0.05 * d)
    assert (d * b).sum() / b.sum() < (d * f).sum() - 1.5


def test_run_scenario_returns_each_method():
    rows = sim.run_scenario("A", reps=2, cutoffs=[179])
    assert {r["method"] for r in rows} == set(sim.METHODS)
    assert all(np.isfinite(r["relbias_mean"]) for r in rows)


def test_impute_linelist_em_recovers_onsets():
    rng = np.random.default_rng(1)
    name = "C"
    sp = sim.scenario_setup(name)
    onset, report, missing = sim.simulate(name, sp, rng)
    T = 140
    keep = report <= T
    onset, report, missing = onset[keep], report[keep], missing[keep]
    obs = np.where(missing, np.nan, onset.astype(float))
    res = sim.impute_linelist(obs, report, cutoff=T)
    truth = np.bincount(onset, minlength=T + 1)
    week = slice(56, 63)  # a busy week before the turn
    rel = res["imputed"]["em"][week].sum() / truth[week].sum() - 1
    assert abs(rel) < 0.1
    assert abs(res["imputed"]["em"].sum() - keep.sum()) < 1e-6 * keep.sum()
