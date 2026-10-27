# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
import numpy as np
import pytest

from onset_imputation.delays import DMAX, F_DEFAULT
from onset_imputation.scenarios import (
    DESC,
    SCEN,
    Scenario,
    miss_prob,
    r_path,
    scenario_setup,
    simulate,
)


def test_every_scenario_has_a_description():
    assert set(SCEN) == set(DESC)


def test_scenario_settings_unpack_as_tuples():
    days, target, cp = SCEN["C"]
    assert (days, target, cp) == (150, 5000, 12)
    assert SCEN["C"] == Scenario(150, 5000, 12)
    assert SCEN["C"].change_point_week == 12


def test_scenario_tables_are_read_only():
    with pytest.raises(TypeError):
        SCEN["Z"] = Scenario(10, 10, None)  # type: ignore[index]
    with pytest.raises(TypeError):
        DESC["Z"] = "new"  # type: ignore[index]


@pytest.mark.parametrize("name", list(SCEN))
def test_scenario_setup_shapes_and_totals(name):
    setup = scenario_setup(name)
    days, target, cp = SCEN[name]
    assert setup["L"] == days
    assert setup["cp"] == cp
    assert setup["O"].shape == (days,)
    assert np.isclose(setup["O"].sum(), target)
    assert setup["fmat"].shape == (days, DMAX + 1)
    assert np.allclose(setup["fmat"].sum(1), 1)
    assert r_path(name, days).shape == (days,)


def test_scenario_f_delay_shortens():
    fmat = scenario_setup("F")["fmat"]
    d = np.arange(DMAX + 1)
    assert fmat[0] @ d > fmat[-1] @ d + 5


def test_scenario_i_doubles_from_day_70():
    expected = scenario_setup("I")["O"]
    growth = np.exp(r_path("I", 150)[70])
    assert np.isclose(expected[70] / expected[69], 2 * growth)


def test_other_scenarios_use_default_delay():
    assert np.allclose(scenario_setup("A")["fmat"], F_DEFAULT)


@pytest.mark.parametrize("func", [scenario_setup, lambda n: r_path(n, 10)])
def test_unknown_scenario_is_a_clear_error(func):
    with pytest.raises(ValueError, match="unknown scenario 'Z'"):
        func("Z")


def test_miss_prob_by_scenario():
    delay = np.array([0, 10, 40])
    report = np.array([50, 80, 120])
    assert np.allclose(miss_prob("A", delay, report), 0.4)
    assert np.allclose(miss_prob("G", delay, report), [0.2, 0.45, 0.7])
    assert np.allclose(miss_prob("H", delay, report), [0.2, 0.4, 0.6])


@pytest.mark.parametrize("name", ["A", "F", "G"])
def test_simulate_is_consistent_and_reproducible(name):
    setup = scenario_setup(name)
    onset, report, missing = simulate(name, setup, np.random.default_rng(2))
    again = simulate(name, setup, np.random.default_rng(2))
    for x, y in zip((onset, report, missing), again, strict=True):
        assert np.array_equal(x, y)
    delay = report - onset
    assert onset.size == report.size == missing.size
    assert np.all((delay >= 0) & (delay <= DMAX))
    assert np.all((onset >= 0) & (onset < setup["L"]))
    assert missing.dtype == bool
    assert 0.2 < missing.mean() < 0.7


def test_simulate_rejects_unknown_scenario():
    setup = scenario_setup("A")
    with pytest.raises(ValueError, match="unknown scenario"):
        simulate("Z", setup, np.random.default_rng(1))
