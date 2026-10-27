# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Simulation scenarios: settings, expected curves and simulated line lists.

Days are integers.
Day 0 is the first possible onset in a scenario.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray

from .delays import DMAX, F_DEFAULT, gamma_pmf

__all__ = [
    "DESC",
    "SCEN",
    "Scenario",
    "ScenarioSetup",
    "miss_prob",
    "r_path",
    "scenario_setup",
    "simulate",
]


class Scenario(NamedTuple):
    """Settings of one scenario.

    A named tuple, so `L, target, cp = SCEN[name]` still unpacks it.

    Attributes
    ----------
    days : int
        Number of onset days, `L`.
    target : int
        Expected total number of cases.
    change_point_week : int or None
        Change-point week used for the `*_cp` summary columns, or `None`.
    """

    days: int
    target: int
    change_point_week: int | None


SCEN: Final[Mapping[str, Scenario]] = MappingProxyType(
    {
        "A": Scenario(120, 10000, None),
        "B": Scenario(150, 5000, 8),  # switch at day 60
        "B0": Scenario(150, 4000, None),
        "C": Scenario(150, 5000, 12),  # r crosses 0 at day ~89
        "D": Scenario(150, 5000, 11),  # switch at day 80
        "E": Scenario(150, 3000, None),
        "F": Scenario(150, 5000, None),
        "G": Scenario(150, 5000, 12),
        "H": Scenario(150, 5000, 12),
        "I": Scenario(150, 5000, 10),  # ascertainment doubles at day 70
        "Csmall": Scenario(150, 300, 12),
        "W": Scenario(150, 5000, 7),  # first turn at day ~40
    }
)
"""Scenario settings, keyed by name. Read-only.

Each value is a `Scenario` `(L, target, cp)`: the number of onset days,
the expected total number of cases and the change-point week (or
`None`).
The growth rate path is set in `r_path` and missingness in `miss_prob`.
See `DESC`.
"""

DESC: Final[Mapping[str, str]] = MappingProxyType(
    {
        "A": "r=0.05 constant",
        "B": "r=0.05 then -0.03 at d60",
        "B0": "r=0 stable",
        "C": "r 0.05->-0.02 over d60-100",
        "D": "r 0.06->-0.04 at d80",
        "E": "r=0.008 (Isiro-like)",
        "F": "r=0.03, mean delay 12->4",
        "G": "as C, P(miss) 0.2->0.7 with delay",
        "H": "as C, P(miss) 0.2->0.6 by report d60-100",
        "I": "as C, ascertainment x2 at d70",
        "Csmall": "as C, ~300 cases",
        "W": "two waves: growth, control, resurgence, control",
    }
)
"""One-line description of each scenario in `SCEN`. Read-only.

Unless stated, the delay is `F_DEFAULT` and each onset is missing with
probability 0.4.
"""


class ScenarioSetup(TypedDict):
    """Scenario inputs returned by `scenario_setup`.

    The single-letter keys match the notation in the module docs and are
    kept for compatibility.

    Attributes
    ----------
    L : int
        Number of onset days.
    O : numpy.ndarray of float
        Expected daily onsets, shape (L,).
    fmat : numpy.ndarray of float
        Forward delay pmf by onset day, shape (L, DMAX + 1).
    cp : int or None
        Change-point week, or `None`.
    """

    L: int
    O: NDArray[np.float64]  # noqa: E741 - key kept for compatibility
    fmat: NDArray[np.float64]
    cp: int | None


def _check_scenario(name: str) -> Scenario:
    """Return the settings of scenario `name` or raise a clear error."""
    try:
        return SCEN[name]
    except KeyError:
        known = ", ".join(SCEN)
        raise ValueError(
            f"unknown scenario {name!r}; choose one of {known}"
        ) from None


def r_path(name: str, days: int) -> NDArray[np.float64]:
    """Compute the daily growth rate path of a scenario.

    Parameters
    ----------
    name : str
        A key of `SCEN`.
    days : int
        Number of onset days, `L`.

    Returns
    -------
    numpy.ndarray of float
        Growth rate r(t) per day for days 0..days-1.

    Raises
    ------
    ValueError
        If `name` is not a known scenario.
    """
    t = np.arange(days, dtype=np.float64)
    if name == "A":
        return np.full(days, 0.05)
    if name == "B":
        return np.where(t < 60, 0.05, -0.03)
    if name == "B0":
        return np.zeros(days)
    if name in ("C", "G", "H", "I", "Csmall"):
        return np.interp(t, [0, 60, 100, days], [0.05, 0.05, -0.02, -0.02])
    if name == "D":
        return np.where(t < 80, 0.06, -0.04)
    if name == "E":
        return np.full(days, 0.008)
    if name == "F":
        return np.full(days, 0.03)
    if name == "W":
        return np.interp(
            t,
            [0, 30, 50, 90, 105, 135, 150],
            [0.06, 0.06, -0.03, -0.03, 0.05, 0.05, -0.04],
        )
    _check_scenario(name)
    raise ValueError(f"no growth rate path for scenario {name!r}")


def scenario_setup(name: str) -> ScenarioSetup:
    """Build the expected onset curve and delays for a scenario.

    Parameters
    ----------
    name : str
        A key of `SCEN`.

    Returns
    -------
    ScenarioSetup
        Scenario inputs with keys:

        - `L`: number of onset days.
        - `O`: expected daily onsets, shape (L,), summing to the target
          total. Built as exp(cumulative sum of r(t)) and rescaled. In
          scenario `I` it is doubled from onset day 70 before rescaling.
        - `fmat`: forward delay pmf by onset day, shape (L, DMAX + 1).
          In scenario `F` the mean (and SD) falls linearly from 12 to 4
          days; otherwise every row is `F_DEFAULT`.
        - `cp`: change-point week or `None`.

    Raises
    ------
    ValueError
        If `name` is not a known scenario.
    """
    days, target, change_point = _check_scenario(name)
    growth = r_path(name, days)
    shape = np.exp(np.concatenate([[0], np.cumsum(growth[1:])]))
    if name == "I":
        shape = shape * np.where(np.arange(days) >= 70, 2.0, 1.0)
    expected_onsets = shape * target / shape.sum()
    if name == "F":
        means = np.linspace(12, 4, days)
        fmat = np.array([gamma_pmf(m, m) for m in means])
    else:
        fmat = np.tile(F_DEFAULT, (days, 1))
    return {"L": days, "O": expected_onsets, "fmat": fmat, "cp": change_point}


def miss_prob(
    name: str, delay: NDArray[np.int64], report: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Compute the probability that each case's onset is missing.

    The probability is 0.4 for all cases, except in scenario `G` (0.2
    rising to 0.7 as the delay goes from 0 to 20 days) and `H` (0.2
    rising to 0.6 as the report day goes from 60 to 100).

    Parameters
    ----------
    name : str
        A key of `SCEN`.
    delay : numpy.ndarray of int
        Onset-to-report delay of each case, in days.
    report : numpy.ndarray of int
        Report day of each case. Same shape as `delay`.

    Returns
    -------
    numpy.ndarray of float
        Probability that each case's onset is missing, same shape as
        `delay`.
    """
    if name == "G":
        return 0.2 + 0.5 * np.clip(delay / 20, 0, 1)
    if name == "H":
        return np.interp(report, [0, 60, 100, 1e9], [0.2, 0.2, 0.6, 0.6])
    return np.full(delay.shape, 0.4)


def simulate(
    name: str, setup: ScenarioSetup, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    """Simulate one line list for a scenario.

    Daily onset counts are Poisson with mean `setup["O"]`.
    Each case gets a delay from its onset day's forward pmf, a report
    day of onset plus delay, and a missing-onset flag from `miss_prob`.

    Parameters
    ----------
    name : str
        A key of `SCEN`.
    setup : ScenarioSetup
        Output of `scenario_setup(name)`.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    onset : numpy.ndarray of int
        Onset day of each case.
    report : numpy.ndarray of int
        Report day of each case. No cut-off is applied, so reports run
        to `L - 1 + DMAX`.
    missing : numpy.ndarray of bool
        Whether each case's onset is missing.

    Raises
    ------
    ValueError
        If `name` is not a known scenario.
    """
    _check_scenario(name)
    counts = rng.poisson(setup["O"])
    onset = np.repeat(np.arange(setup["L"]), counts)
    if name == "F":
        cdfs = np.cumsum(setup["fmat"], axis=1)
        u = rng.random(onset.size)
        delay = (u[:, None] > cdfs[onset]).sum(axis=1)
        delay = np.minimum(delay, DMAX)
    else:
        delay = rng.choice(DMAX + 1, size=onset.size, p=F_DEFAULT)
    report = onset + delay
    missing = rng.random(onset.size) < miss_prob(name, delay, report)
    return onset, report, missing
