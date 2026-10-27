import numpy as np

import imputation_sim as sim
from imputation_sim.simulation import DMAX


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
