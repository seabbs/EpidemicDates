# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Simulation study: bias of onset-date imputation for missing onsets.

`run_scenario` simulates replicate line lists for one scenario, imputes
them with each of `METHODS` at each cut-off and summarises the bias.
`run_study` does this for several scenarios and `write_results` saves
the rows.
"""

from __future__ import annotations

import csv
import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .delays import DMAX, F_DEFAULT
from .methods import METHODS, em_backproject, round_half_up
from .scenarios import DESC, SCEN, ScenarioSetup, scenario_setup, simulate

__all__ = [
    "OFF",
    "analytic",
    "first_negative_week",
    "growth_rate",
    "run_rep",
    "run_scenario",
    "run_study",
    "summarise",
    "weekly",
    "write_results",
]

OFF: Final[int] = 70
"""Day offset so negative imputed onsets bin correctly into weeks."""

Replicate = dict[int, dict[str, Any]]
"""Output of `run_rep`: results for one replicate, keyed by cut-off."""

SummaryRow = dict[str, Any]
"""One summary row from `summarise`, keyed by column name."""


def weekly(
    days: ArrayLike,
    weights: ArrayLike | None = None,
    n_weeks: int = 40,
) -> NDArray[Any]:
    """Sum (weighted) days into weeks 0..n_weeks-1.

    Week 0 is days 0 to 6.

    Parameters
    ----------
    days : array_like of int
        Day of each case. Must be at least `-OFF`.
    weights : array_like of float, optional
        Weight of each entry of `days`. Defaults to 1 each.
    n_weeks : int, optional
        Number of weeks returned.

    Returns
    -------
    numpy.ndarray
        Weekly totals, length `n_weeks`. Integer if `weights` is `None`,
        float otherwise. Days before 0 or after the last week are
        discarded.
    """
    idx = (np.asarray(days) + OFF) // 7
    minlength = n_weeks + OFF // 7
    totals: NDArray[Any]
    if weights is None:
        totals = np.bincount(idx, minlength=minlength)
    else:
        totals = np.bincount(
            idx,
            weights=np.asarray(weights, dtype=np.float64),
            minlength=minlength,
        )
    return totals[OFF // 7 : OFF // 7 + n_weeks]


def run_rep(
    name: str,
    setup: ScenarioSetup,
    rng: np.random.Generator,
    cutoffs: Sequence[int],
    forward_pmf_by_day: NDArray[np.float64] | None,
) -> Replicate:
    """Simulate one line list and impute it at each cut-off.

    Parameters
    ----------
    name : str
        A key of `SCEN`.
    setup : ScenarioSetup
        Output of `scenario_setup(name)`.
    rng : numpy.random.Generator
        Random number generator.
    cutoffs : sequence of int
        Data cut-off days.
    forward_pmf_by_day : numpy.ndarray of float or None
        Forward delay pmf by day, shape (L, DMAX + 1), for the `forward`
        method, or `None` to use `F_DEFAULT`. The row used for a missing
        case is its report day (capped at L - 1). In scenario `F`,
        delays are simulated by onset day, so this is an approximation
        of the oracle rather than an exact oracle.

    Returns
    -------
    dict of int to dict
        Keyed by cut-off. Each value holds weekly onset counts for
        `truth` (cases reported by the cut-off) and each of `METHODS`,
        plus `n`, the number of cases reported by the cut-off.
    """
    onset, report, missing = simulate(name, setup, rng)
    out: Replicate = {}
    for cutoff in cutoffs:
        reported = report <= cutoff
        on, rep, mi = onset[reported], report[reported], missing[reported]
        res: dict[str, Any] = {"truth": weekly(on)}
        comp_on = on[~mi]
        comp_d = (rep - on)[~mi]
        rm = rep[mi]
        base = weekly(comp_on)
        res["backsampling"] = base + weekly(
            rm - rng.choice(comp_d, size=rm.size)
        )
        res["mean"] = base + weekly(rm - round_half_up(comp_d.mean()))
        if forward_pmf_by_day is None:
            fd = rng.choice(DMAX + 1, size=rm.size, p=F_DEFAULT)
        else:
            # time-varying forward pmf, indexed by report day
            fm = forward_pmf_by_day[np.minimum(rm, setup["L"] - 1)]
            u = rng.random(rm.size)
            fd = np.minimum((u[:, None] > np.cumsum(fm, 1)).sum(1), DMAX)
        res["forward"] = base + weekly(rm - fd)
        alloc = em_backproject((comp_on, comp_d), rm, cutoff)
        res["em"] = base + weekly(np.arange(cutoff + 1), weights=alloc)
        res["n"] = reported.sum()
        out[cutoff] = res
    return out


def growth_rate(counts: NDArray[Any]) -> NDArray[np.float64]:
    """Compute the centred 3-week log-linear growth rate per day.

    Parameters
    ----------
    counts : numpy.ndarray
        Weekly counts, with weeks on the last axis.

    Returns
    -------
    numpy.ndarray of float
        Growth rate per day, `(log(y[w + 1] + 0.5) - log(y[w - 1] + 0.5))
        / 14`, same shape as `counts`. NaN in the first and last weeks.
    """
    log_counts = np.log(counts + 0.5)
    rate = np.full(counts.shape, np.nan)
    rate[..., 1:-1] = (log_counts[..., 2:] - log_counts[..., :-2]) / 2 / 7
    return rate


def first_negative_week(
    rates: NDArray[np.float64], start: int = 4
) -> NDArray[np.float64]:
    """Find the first week with two consecutive negative growth rates.

    Parameters
    ----------
    rates : numpy.ndarray of float
        Growth rates, shape (replicates, weeks).
    start : int, optional
        First week searched.

    Returns
    -------
    numpy.ndarray of float
        One week per row of `rates`, NaN if there is none.
    """
    out = np.full(rates.shape[0], np.nan)
    for i in range(rates.shape[0]):
        for week in range(start, rates.shape[1] - 2):
            if rates[i, week] < 0 and rates[i, week + 1] < 0:
                out[i] = week
                break
    return out


def summarise(
    name: str,
    setup: ScenarioSetup,
    cutoff: int,
    replicates: Sequence[Replicate],
    retro: bool,
) -> list[SummaryRow]:
    """Summarise each method's bias over replicates at one cut-off.

    Parameters
    ----------
    name : str
        A key of `SCEN`.
    setup : ScenarioSetup
        Output of `scenario_setup(name)`.
    cutoff : int
        Cut-off day.
    replicates : sequence of dict
        Outputs of `run_rep`, each including `cutoff`.
    retro : bool
        Whether `cutoff` is the retrospective cut-off. If so, weeks
        0..L//7-1 are scored. Otherwise the complete weeks up to the
        cut-off are scored.

    Returns
    -------
    list of dict
        One row per method. See `run_scenario` for the columns.
    """
    with warnings.catch_warnings():
        # Weeks at the edges have all-NaN growth rates by design.
        warnings.filterwarnings(
            "ignore", "Mean of empty slice|All-NaN slice", RuntimeWarning
        )
        return _summarise(name, setup, cutoff, replicates, retro)


def _summarise(
    name: str,
    setup: ScenarioSetup,
    cutoff: int,
    replicates: Sequence[Replicate],
    retro: bool,
) -> list[SummaryRow]:
    """Compute the rows for `summarise`, which handles warnings."""
    n_onset_days, cp = setup["L"], setup["cp"]
    nw = (n_onset_days // 7) if retro else (cutoff + 1) // 7
    truth = np.array([r[cutoff]["truth"][:nw] for r in replicates])
    ntot = np.mean([r[cutoff]["n"] for r in replicates])
    mt = truth.mean(0)
    thr = 3 if name == "Csmall" else 10
    inc = mt >= thr
    wk = np.arange(nw)
    cpw = inc & (np.abs(wk - cp) <= 3) if cp is not None else None
    gt = growth_rate(truth)
    rows: list[SummaryRow] = []
    for m in METHODS:
        est = np.array([r[cutoff][m][:nw] for r in replicates])
        rb = est.mean(0) / np.where(mt > 0, mt, np.nan) - 1
        rel_rep = (est - truth) / np.maximum(truth, 1)
        rmse = np.sqrt(np.nanmean(rel_rep[:, inc] ** 2))
        ge = growth_rate(est)
        gb = np.nanmean(ge - gt, axis=0)
        ginc = inc.copy()
        ginc[[0, -1]] = False
        row: SummaryRow = {
            "scenario": name,
            "desc": DESC[name],
            "cutoff": cutoff,
            "retro": int(retro),
            "method": m,
            "mean_cases": round(ntot),
            "relbias_mean": np.nanmean(rb[inc]),
            "relbias_maxabs": np.nanmax(np.abs(rb[inc])),
            "rel_rmse": rmse,
            "gr_bias_meanabs": np.nanmean(np.abs(gb[ginc])),
            "gr_bias_maxabs": np.nanmax(np.abs(gb[ginc])),
        }
        if cpw is not None and cpw.any():
            row["relbias_cp"] = np.nanmean(rb[cpw])
            g_cp = cpw & ginc
            row["gr_bias_cp"] = np.nanmean(gb[g_cp])
            row["gr_bias_cp_maxabs"] = np.nanmax(np.abs(gb[g_cp]))
        if retro:
            pk_t = truth.argmax(1)
            pk_e = est.argmax(1)
            row["peak_shift_wk"] = np.mean(pk_e - pk_t)
            row["peak_diff_frac"] = np.mean(pk_e != pk_t)
            if cp is not None:
                zt = first_negative_week(gt)
                ze = first_negative_week(ge)
                ok = ~np.isnan(zt) & ~np.isnan(ze)
                row["zero_cross_lag_wk"] = np.mean(ze[ok] - zt[ok])
            # mean curve peak
            row["peak_meancurve_shift"] = int(
                np.argmax(est.mean(0)) - np.argmax(mt)
            )
        else:
            row["relbias_last2"] = np.nanmean(rb[nw - 2 : nw])
            row["relbias_last4"] = np.nanmean(rb[nw - 4 : nw])
            row["gr_bias_last"] = gb[nw - 2]
        rows.append(row)
    return rows


def analytic() -> list[str]:
    """Run analytic checks of pooled complete-case delays.

    Compares the forward delay pmf with the backward pmf under constant
    growth, and checks the result by simulation.

    Returns
    -------
    list of str
        Lines of text:

        - for r = 0.05, 0.008 and -0.03, the forward and backward mean
          delays, the retrospective level factor of `backsampling`
          imputed cases,
          E[exp(-rD)] E[exp(rD)], and of `mean` imputed cases,
          exp(rm) E[exp(-rD)], with D ~ `F_DEFAULT` and m its mean
          rounded with halves up;
        - the pooled complete-case mean delay in one simulation of
          scenario `A` at cut-offs 90 and 179 (seed 1);
        - the mean of the analytic pooled right-truncated pmf under
          r = 0.05 from day 0 with T = 90, and its total variation
          distance to the backward pmf.
    """
    f = F_DEFAULT
    d = np.arange(DMAX + 1)
    lines = []
    for r in (0.05, 0.008, -0.03):
        mgf_neg = (f * np.exp(-r * d)).sum()
        mgf_pos = (f * np.exp(r * d)).sum()
        backward = f * np.exp(-r * d) / mgf_neg
        shift = round_half_up(float(np.dot(d, f)))
        lines.append(
            f"r={r:+.3f}: forward mean {np.dot(d, f):.2f}, backward mean "
            f"{np.dot(d, backward):.2f}; retrospective back-sampling level "
            "factor for "
            f"imputed cases E[e^-rD]E[e^rD]={mgf_neg * mgf_pos:.3f}; "
            f"mean-shift factor e^(r m)E[e^-rD]="
            f"{np.exp(r * shift) * mgf_neg:.3f}"
        )
    # empirical check under scenario A
    rng = np.random.default_rng(1)
    setup = scenario_setup("A")
    on, rep, mi = simulate("A", setup, rng)
    for cutoff in (90, 179):
        k = (rep <= cutoff) & ~mi
        dd = rep[k] - on[k]
        lines.append(
            f"Scenario A sim, T={cutoff}: pooled complete-case mean delay "
            f"{dd.mean():.2f}"
        )
    # pooled truncated pmf for growth since day 0, T = 90
    cutoff, r = 90, 0.05
    w = np.array([(np.exp(r * np.arange(0, cutoff - x + 1))).sum() for x in d])
    p = f * w / (f * w).sum()
    backward = f * np.exp(-r * d) / (f * np.exp(-r * d)).sum()
    lines.append(
        f"Analytic pooled truncated pmf (A, T=90) mean "
        f"{np.dot(d, p):.2f}; TV distance to backward "
        f"{0.5 * np.abs(p - backward).sum():.4f}"
    )
    return lines


def run_scenario(
    name: str,
    reps: int = 200,
    seed: int = 20261027,
    cutoffs: Sequence[int] | None = None,
) -> list[SummaryRow]:
    """Simulate one scenario and summarise each method's bias.

    Parameters
    ----------
    name : str
        A key of `SCEN` (see `DESC`).
    reps : int, optional
        Number of stochastic replicates. Must be at least 1.
    seed : int, optional
        Seed for `numpy.random.default_rng`.
    cutoffs : sequence of int, optional
        Data cut-off days, each non-negative. The retrospective cut-off
        is `L - 1 + DMAX`, when every case has been reported. Defaults
        to the retrospective cut-off plus real-time cut-offs 90 and 120
        (70 and 90 for `A`). A cut-off is scored as retrospective only
        if it equals that day.

    Returns
    -------
    list of dict
        One row per cut-off and method. Bias is against the true onsets
        of cases reported by the cut-off. Weeks are scored only if the
        mean true count is at least 10 (3 for `Csmall`).

        Always present:

        - `scenario`, `desc`, `cutoff`, `retro` (1 if retrospective),
          `method`, `mean_cases` (mean cases reported by the cut-off).
        - `relbias_mean`, `relbias_maxabs`: mean and maximum absolute
          relative bias of the replicate-mean weekly count.
        - `rel_rmse`: root mean square relative error of weekly counts
          over replicates and weeks (denominator at least 1).
        - `gr_bias_meanabs`, `gr_bias_maxabs`: mean and maximum absolute
          bias in the weekly growth rate (per day). The growth rate is
          the centred log-linear slope over three weeks of
          log(count + 0.5).

        If the scenario has a change point within the scored weeks:

        - `relbias_cp`, `gr_bias_cp`, `gr_bias_cp_maxabs`: relative bias,
          growth rate bias and its maximum absolute value within three
          weeks of the change point.

        Retrospective rows only:

        - `peak_shift_wk`: mean shift of the peak week (estimate minus
          truth).
        - `peak_diff_frac`: fraction of replicates with a different peak
          week.
        - `zero_cross_lag_wk`: mean lag of the first week (from week 4)
          with two consecutive negative growth rates (scenarios with a
          change point only).
        - `peak_meancurve_shift`: peak shift of the replicate-mean curve.

        Real-time rows only:

        - `relbias_last2`, `relbias_last4`: relative bias over the last
          two and four complete weeks.
        - `gr_bias_last`: growth rate bias in the last week with a
          centred estimate.

    Raises
    ------
    ValueError
        If `name` is not a known scenario, `reps` is below 1 or a
        cut-off is negative.

    Notes
    -----
    In scenario `F` the delay changes with onset day. The `forward`
    method draws from the pmf of the case's report day, as the onset day
    is unknown, so it is close to but not exactly an oracle there.
    """
    if name not in SCEN:
        raise ValueError(
            f"unknown scenario {name!r}; choose one of {', '.join(SCEN)}"
        )
    if reps < 1:
        raise ValueError(f"reps must be at least 1, got {reps}")
    setup = scenario_setup(name)
    retro_t = setup["L"] - 1 + DMAX
    if cutoffs is None:
        cutoffs = [retro_t, 90, 120] if name != "A" else [retro_t, 70, 90]
    if any(c < 0 for c in cutoffs):
        raise ValueError(f"cutoffs must be non-negative, got {list(cutoffs)}")
    rng = np.random.default_rng(seed)
    fwd = setup["fmat"] if name == "F" else None
    sims = [run_rep(name, setup, rng, cutoffs, fwd) for _ in range(reps)]
    rows: list[SummaryRow] = []
    for cutoff in cutoffs:
        rows += summarise(name, setup, cutoff, sims, cutoff == retro_t)
    return rows


def run_study(
    reps: int = 200,
    seed: int = 20261027,
    scenarios: Sequence[str] | None = None,
    verbose: bool = True,
) -> list[SummaryRow]:
    """Run every scenario, or those named, and return all summary rows.

    Parameters
    ----------
    reps : int, optional
        Number of replicates per scenario. Must be at least 1.
    seed : int, optional
        Base seed. Scenario `name` uses `seed + i`, where `i` is the
        position of `name` in `SCEN`, so a scenario gives the same
        results whichever other scenarios are run with it.
    scenarios : sequence of str, optional
        Keys of `SCEN`. Defaults to all scenarios (also when empty).
    verbose : bool, optional
        Print a line as each scenario finishes.

    Returns
    -------
    list of dict
        Rows from `run_scenario` for each scenario, in the order given.

    Raises
    ------
    ValueError
        If a scenario name is unknown or `reps` is below 1.
    """
    names = list(scenarios) if scenarios else list(SCEN)
    unknown = [n for n in names if n not in SCEN]
    if unknown:
        raise ValueError(
            f"unknown scenario(s) {unknown}; choose from {', '.join(SCEN)}"
        )
    order = list(SCEN)
    rows: list[SummaryRow] = []
    for name in names:
        rows += run_scenario(name, reps=reps, seed=seed + order.index(name))
        if verbose:
            print(f"done {name}", flush=True)
    return rows


def write_results(
    rows: Sequence[SummaryRow], out: str | os.PathLike[str]
) -> None:
    """Write summary rows and analytic checks to a directory.

    Parameters
    ----------
    rows : sequence of dict
        Summary rows, for example from `run_study`.
    out : str or os.PathLike
        Output directory. Created if needed.

    Notes
    -----
    Writes `out/results.csv`, with columns in order of first appearance
    and floats to 4 decimal places, and `out/analytic.txt`, the output
    of `analytic`. Existing files are overwritten.
    """
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    k: (f"{v:.4f}" if isinstance(v, float) else v)
                    for k, v in r.items()
                }
            )
    (out_dir / "analytic.txt").write_text(
        "\n".join(analytic()) + "\n", encoding="utf-8"
    )
