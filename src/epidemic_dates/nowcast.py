# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Nowcast onsets from a line list, with uncertainty.

`nowcast_linelist` imputes missing onsets, corrects for right truncation
and samples to give quantiles in the long format forecast hubs use.
`write_nowcast_csv` writes the quantile table as CSV.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import os
from collections.abc import Sequence
from typing import Final, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .delays import DMAX
from .linelist import _prepare
from .methods import (
    DELAY_SHRINKAGE,
    EMFit,
    _check_shrinkage,
    _em_iterate,
    as_delay,
    backsampling_allocate,
    delay_periods,
    em_fit,
    mean_shift_allocate,
    posterior_allocate,
    round_half_up,
)

__all__ = [
    "NOWCAST_METHODS",
    "QUANTILES",
    "NowcastResult",
    "nowcast_linelist",
    "write_nowcast_csv",
]

NOWCAST_METHODS: Final[tuple[str, ...]] = (
    "em",
    "backsampling",
    "mean",
    "forward",
)
"""Imputation methods `nowcast_linelist` accepts. `em` is the default."""

QUANTILES: Final[tuple[float, ...]] = (
    0.025,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.975,
)
"""Default quantile levels: the median and 50%, 80% and 95% intervals."""


class NowcastResult(TypedDict):
    """Output of `nowcast_linelist`.

    Daily arrays have length T + 1 and index i is onset day i.
    Weekly arrays have one entry per full week ending on the cut-off.
    Days and weeks without an estimate hold NaN.

    Attributes
    ----------
    method : str
        Imputation method used.
    cutoff : int
        Cut-off day T.
    dropped : int
        Number of cases dropped before imputation.
    delay : numpy.ndarray of float
        Forward delay pmf (0..DMAX) for the point nowcast: the given
        static `delay`, the EM estimate, or with periods or a delay
        matrix the pooled or onset-weighted mean delay.
    delay_by_period : numpy.ndarray of float
        Delay pmf of each period, one row per period. Each onset day is
        its own period when a delay matrix is given.
    period_start : numpy.ndarray of int
        First onset day of each period.
    reported : numpy.ndarray of float
        Fraction of each onset day's cases reported by the cut-off,
        $F_{k(s)}(T - s)$, under the delay of the day's period.
    point : numpy.ndarray of float
        Point nowcast by onset day.
    point_weekly : numpy.ndarray of float
        Point nowcast by week.
    week_start : numpy.ndarray of int
        First onset day of each week.
    samples : numpy.ndarray of float
        Sampled nowcasts, shape `(n_samples, T + 1)`.
    weekly_samples : numpy.ndarray of float
        Sampled weekly nowcasts, shape `(n_samples, n_weeks)`.
    quantiles : tuple of float
        Quantile levels.
    daily_quantiles : list of dict
        Rows with `target_end_date`, `quantile` and `value`.
    weekly_quantiles : list of dict
        Rows with `week_start`, `target_end_date`, `quantile` and
        `value`.
    """

    method: str
    cutoff: int
    dropped: int
    delay: NDArray[np.float64]
    delay_by_period: NDArray[np.float64]
    period_start: NDArray[np.int64]
    reported: NDArray[np.float64]
    point: NDArray[np.float64]
    point_weekly: NDArray[np.float64]
    week_start: NDArray[np.int64]
    samples: NDArray[np.float64]
    weekly_samples: NDArray[np.float64]
    quantiles: tuple[float, ...]
    daily_quantiles: list[dict[str, object]]
    weekly_quantiles: list[dict[str, object]]


def _complete_matrix(
    onset_c: NDArray[np.int64], delay_c: NDArray[np.int64], n_days: int
) -> NDArray[np.float64]:
    """Count complete cases by onset day and delay."""
    out = np.zeros((n_days, DMAX + 1))
    np.add.at(out, (onset_c, delay_c), 1.0)
    return out


def _weekly(daily: NDArray[np.float64], n_weeks: int) -> NDArray[np.float64]:
    """Sum the last `7 * n_weeks` days into weeks ending on the last day.

    A week with any NaN day is NaN. Works along the last axis.
    """
    tail = daily[..., daily.shape[-1] - 7 * n_weeks :]
    return np.asarray(
        tail.reshape(*daily.shape[:-1], n_weeks, 7).sum(-1), dtype=np.float64
    )


def _draw_backwards(
    missing: NDArray[np.float64],
    pmf: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Give each missing case an onset of report day minus a drawn delay.

    Onsets before day 0 are dropped, as in `backsampling_allocate`.
    """
    n_days = missing.size
    out = np.zeros(n_days)
    total = pmf.sum()
    if total <= 0:
        return out
    prob = pmf / total
    for t in np.flatnonzero(missing):
        draws = rng.multinomial(int(missing[t]), prob)
        d = np.arange(min(prob.size, t + 1))
        out[t - d] += draws[: d.size]
    return out


def _draw_em(
    missing: NDArray[np.float64],
    onsets: NDArray[np.float64],
    delay_by_day: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    r"""Draw onsets of missing cases from the EM posterior.

    A case reported on day $t$ has onset $s$ with probability
    proportional to $\lambda(s) f_s(t - s)$, with $f_s$ row $s$ of
    `delay_by_day`.
    """
    out = np.zeros(missing.size)
    n_delays = delay_by_day.shape[1]
    for t in np.flatnonzero(missing):
        d = np.arange(min(n_delays, t + 1))
        w = onsets[t - d] * delay_by_day[t - d, d]
        if w.sum() <= 0:
            continue
        out[t - d] += rng.multinomial(int(missing[t]), w / w.sum())
    return out


def _quantile_rows(
    values: NDArray[np.float64],
    levels: Sequence[float],
    end_dates: Sequence[object],
    start_dates: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    """Long table of quantiles, skipping columns without an estimate."""
    rows: list[dict[str, object]] = []
    for j in range(values.shape[1]):
        col = values[:, j]
        if not np.all(np.isfinite(col)):
            continue
        qs = np.quantile(col, levels)
        for level, value in zip(levels, qs, strict=True):
            row: dict[str, object] = {}
            if start_dates is not None:
                row["week_start"] = start_dates[j]
            row["target_end_date"] = end_dates[j]
            row["quantile"] = float(level)
            row["value"] = float(value)
            rows.append(row)
    return rows


def nowcast_linelist(
    onset: ArrayLike,
    report: ArrayLike,
    cutoff: int | None = None,
    method: str = "em",
    delay: ArrayLike | None = None,
    *,
    n_samples: int = 1000,
    seed: int = 1,
    quantiles: Sequence[float] = QUANTILES,
    min_reported: float = 0.05,
    bootstrap: bool = True,
    n_iter: int = 150,
    origin: dt.date | str | None = None,
    delay_period: int | None = None,
    delay_shrinkage: float = DELAY_SHRINKAGE,
) -> NowcastResult:
    r"""Nowcast onsets from a line list, with uncertainty.

    There are three steps.

    **1. Impute.**
    Cases with a known onset keep it.
    Missing onsets are imputed by `method`:

    - `em` (default): EM back-projection (`em_fit`),
      $P(s \mid t) \propto \lambda(s) f(t - s)$;
    - `backsampling`: $P(s \mid t) = g(t - s)$, with $g$ the pooled
      empirical delay of complete cases;
    - `mean`: $s = t - \operatorname{round}\big(\sum_d d \, g(d)\big)$;
    - `forward`: $P(s \mid t) = f(t - s)$.

    Here $f$ is `delay` if given, used as is (a plug-in).
    It can be one pmf or a matrix with one pmf per onset day, so
    $f_s$ can change over time.
    Otherwise it is estimated by `em_fit`, which allows for right
    truncation: one static delay, or with `delay_period` one delay
    $f_k$ per period of onset days, shrunk towards the previous period's delay.
    Write $f_s$ for the delay of onset day $s$ in every case.
    When $f_s$ changes over time, `forward` uses
    $P(s \mid t) \propto f_s(t - s)$ over onset days $0, \dots, t$.
    This gives $\hat n(s) = n_c(s) + n_i(s)$, the onsets on day $s$ of
    cases reported by the cut-off $T$.

    **2. Correct for right truncation.**
    With $F_s$ the CDF of $f_s$, the point nowcast is

    $$\hat C(s) = \frac{\hat n(s)}{F_s\big(\min(T - s, D)\big)}.$$

    Days with $F_s(T - s)$ below `min_reported` get no estimate (NaN).

    **3. Sample.**
    Each of `n_samples` samples does the following, with a generator
    seeded by `seed`.

    - If `bootstrap`, resample the complete cases with replacement.
      Recompute $g$ from them.
      Refit EM to the resampled complete cases and the missing cases,
      warm-started from the point fit for `max(10, n_iter // 5)`
      iterations, to get $f_b$ and $\lambda_b$.
      With `delay_period`, $f_b$ holds one delay per period, refitted
      in the same way.
      With a given `delay`, $f_b$ is that delay and only $\lambda_b$ is
      refitted.
      Without `bootstrap`, $g$, $f_b = f$ and $\lambda_b = \lambda$ are
      fixed.
    - Draw the onsets of the missing cases.
      `backsampling` and `forward` draw $m(t)$ cases reported on day
      $t$ as a multinomial over delays with $g$ or $f_b$.
      `em` draws a multinomial over onset days with
      $P(s \mid t) \propto \lambda_b(s) f_{b,s}(t - s)$.
      With a delay that changes over time, `forward` draws the same way
      with $\lambda_b$ constant.
      `mean` shifts every case by the rounded mean of $g$, with no
      draw.
      This gives $n_b(s)$, which adds the complete cases on day $s$.
    - Draw the cases with onset $s$ not yet reported,

      $$U(s) \sim \operatorname{NegBin}\big(n_b(s), F_{b,s}(T - s)\big),$$

      the number of failures before $n_b(s)$ successes with success
      probability $F_{b,s}(T - s)$, the CDF of the onset day's delay,
      which has mean $n_b(s) \big(1 - F_{b,s}\big) / F_{b,s}$.
      This is the posterior for the total under a prior proportional to
      $1 / C(s)$, given $n_b(s)$ reported with probability $F_b$.
      The sample is $C_b(s) = n_b(s) + U(s)$.

    Weekly samples sum the daily samples over full weeks that end on the
    cut-off.
    A week with any day without an estimate has no estimate.
    Quantiles use linear interpolation (`numpy.quantile`).

    **What the samples capture.**
    Sampling of the missing onsets given the delay and onset curve.
    The binomial reporting of recent onsets, given the delay.
    With `bootstrap`, sampling uncertainty in the delay from the finite
    number of complete cases (and in the onset curve used by `em`).

    **What they do not capture.**
    Error in the model: a delay that changes over time but is estimated
    as static, or that changes within a period; onsets missing other
    than at random given report day; or reporting that is not
    independent between cases.
    Uncertainty in the given `delay`, if one is passed.
    Noise in the onset process itself: the target is the onsets that
    will eventually be reported, not an expected curve.
    Days with no reported onsets get a nowcast of 0 in every sample.

    Parameters
    ----------
    onset : array_like
        Onset day per case, with NaN (or None) where missing.
    report : array_like of int
        Report day per case. Must have no missing values.
    cutoff : int, optional
        Data cut-off day T. Defaults to the last report day.
    method : {"em", "backsampling", "mean", "forward"}, optional
        Imputation method. Defaults to `em`.
    delay : array_like of float, optional
        Forward delay, used as given for the truncation correction, for
        `forward`, and held fixed by `em`. Either one pmf indexed by
        delay in days, or a matrix with one pmf per onset day 0..T in
        its rows. If omitted, it is estimated by `em_fit`. Entries
        beyond `DMAX` are dropped and each pmf renormalised. Cannot be
        combined with `delay_period`.
    n_samples : int, optional
        Number of samples. Must be at least 1.
    seed : int, optional
        Seed for `numpy.random.default_rng`.
    quantiles : sequence of float, optional
        Quantile levels, each in [0, 1]. Defaults to `QUANTILES`.
    min_reported : float, optional
        Smallest fraction reported, $F(T - s)$, for a day to get an
        estimate. Must be in (0, 1].
    bootstrap : bool, optional
        Resample complete cases to include delay uncertainty.
    n_iter : int, optional
        Number of EM iterations for the point fit. Must be at least 1.
    origin : datetime.date or str, optional
        Date of day 0, as a date or an ISO string. If given, dates in
        the quantile tables are ISO strings. Otherwise they are day
        numbers.
    delay_period : int, optional
        Length in days of the periods of onset days that get their own
        estimated delay, counted back from the cut-off. `None` (the
        default) estimates one static delay. See `em_fit`.
    delay_shrinkage : float, optional
        Strength, in expected onsets, of the shrinkage of each period's
        delay towards the previous period's delay. Must be positive.

    Returns
    -------
    NowcastResult
        A dict with the point nowcast, daily and weekly samples, and
        daily and weekly quantile tables in long format (one row per
        date and quantile).

    Raises
    ------
    ValueError
        If the line list is invalid (see `impute_linelist`), `method` is
        unknown, `delay` is not a valid pmf or matrix of pmfs, both
        `delay` and `delay_period` are given, or `n_samples`,
        `quantiles`, `min_reported`, `n_iter`, `delay_period` or
        `delay_shrinkage` are out of range.

    See Also
    --------
    impute_linelist : the point imputations without truncation
        correction.
    write_nowcast_csv : write the quantile table.

    Examples
    --------
    >>> import numpy as np
    >>> onset = np.array([0, 1, np.nan, 3, 4, np.nan, 6, 8, np.nan, 9])
    >>> report = np.array([2, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> res = nowcast_linelist(onset, report, n_samples=50, seed=1)
    >>> res["samples"].shape
    (50, 13)
    >>> res["daily_quantiles"][0]["target_end_date"], len(res["quantiles"])
    (0, 7)
    """
    if method not in NOWCAST_METHODS:
        raise ValueError(
            f"method must be one of {', '.join(NOWCAST_METHODS)}, "
            f"got {method!r}"
        )
    if int(n_samples) != n_samples or n_samples < 1:
        raise ValueError(
            f"n_samples must be a whole number of at least 1, got {n_samples}"
        )
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}")
    if not (math.isfinite(min_reported) and 0 < min_reported <= 1):
        raise ValueError(f"min_reported must be in (0, 1], got {min_reported}")
    levels = tuple(float(q) for q in quantiles)
    if not levels or any(not (0 <= q <= 1) for q in levels):
        raise ValueError("quantiles must be a non-empty sequence in [0, 1]")
    shrinkage = _check_shrinkage(delay_shrinkage)
    if delay is not None and delay_period is not None:
        raise ValueError("give either delay or delay_period, not both")
    if isinstance(origin, str):
        origin = dt.date.fromisoformat(origin)

    prep = _prepare(onset, report, cutoff)
    cutoff = prep.cutoff
    n_days = cutoff + 1
    fixed = None if delay is None else as_delay(delay, cutoff)
    if fixed is not None and fixed.ndim == 2:
        period = np.arange(n_days, dtype=np.int64)
    else:
        period = delay_periods(cutoff, delay_period)
    varying = int(period[-1]) > 0
    n_samples = int(n_samples)
    complete = np.bincount(prep.onset, minlength=n_days)[:n_days].astype(
        np.float64
    )
    missing = np.bincount(prep.missing_report, minlength=n_days)[
        :n_days
    ].astype(np.float64)
    days_left = np.minimum(cutoff - np.arange(n_days), DMAX)

    def empirical(delays: NDArray[np.int64]) -> NDArray[np.float64]:
        counts = np.bincount(delays, minlength=DMAX + 1).astype(np.float64)
        counts /= counts.sum()
        return counts

    need_em = method == "em" or fixed is None
    fit: EMFit | None = None
    if need_em or varying:
        fit = em_fit(
            (prep.onset, prep.delay),
            prep.missing_report,
            cutoff,
            n_iter=n_iter,
            delay=fixed,
            delay_period=delay_period,
            delay_shrinkage=shrinkage,
        )
    if fixed is not None and fixed.ndim == 1:
        f = fixed
        f_periods = fixed[None, :]
        period_start = np.zeros(1, dtype=np.int64)
    else:
        assert fit is not None
        f, f_periods, period_start = (
            fit.delay,
            fit.delay_by_period,
            fit.period_start,
        )
    f_days = f_periods[period]
    g = empirical(prep.delay)
    reported = np.cumsum(f_periods, axis=1)[period, days_left]
    ok = reported >= min_reported

    if method == "em":
        assert fit is not None
        imputed = complete + fit.allocation
    elif method == "backsampling":
        imputed = complete + backsampling_allocate(missing, g)
    elif method == "forward" and varying:
        flat = np.ones(n_days)
        imputed = complete + posterior_allocate(missing, flat, f_days)
    elif method == "forward":
        imputed = complete + backsampling_allocate(missing, f)
    else:
        shift = round_half_up(float(np.dot(np.arange(DMAX + 1), g)))
        imputed = complete + mean_shift_allocate(missing, shift)
    point = np.where(ok, imputed / np.maximum(reported, 1e-12), np.nan)

    rng = np.random.default_rng(seed)
    n_boot_iter = max(10, n_iter // 5)
    samples = np.full((n_samples, n_days), np.nan)
    n_complete = prep.onset.size
    for b in range(n_samples):
        f_b, g_b, f_b_periods = f, g, f_periods
        lam_b = fit.onsets if fit is not None else None
        if bootstrap:
            idx = rng.integers(0, n_complete, n_complete)
            g_b = empirical(prep.delay[idx])
            if need_em:
                assert fit is not None
                refit = _em_iterate(
                    _complete_matrix(prep.onset[idx], prep.delay[idx], n_days),
                    missing,
                    cutoff,
                    fit.onsets,
                    fit.delay_by_period,
                    n_boot_iter,
                    fixed is not None,
                    period,
                    shrinkage,
                )
                lam_b = refit.onsets
                if fixed is None:
                    f_b, f_b_periods = refit.delay, refit.delay_by_period
        if method == "em":
            assert lam_b is not None
            drawn = _draw_em(missing, lam_b, f_b_periods[period], rng)
        elif method == "backsampling":
            drawn = _draw_backwards(missing, g_b, rng)
        elif method == "forward" and varying:
            drawn = _draw_em(missing, np.ones(n_days), f_b_periods[period], rng)
        elif method == "forward":
            drawn = _draw_backwards(missing, f_b, rng)
        else:
            shift = round_half_up(float(np.dot(np.arange(DMAX + 1), g_b)))
            drawn = mean_shift_allocate(missing, shift)
        n_b = complete + drawn
        rep_b = np.clip(
            np.cumsum(f_b_periods, axis=1)[period, days_left], 1e-12, 1.0
        )
        unseen = np.zeros(n_days)
        pos = ok & (n_b > 0) & (rep_b < 1)
        unseen[pos] = rng.negative_binomial(n_b[pos], rep_b[pos])
        samples[b] = np.where(ok, n_b + unseen, np.nan)

    n_weeks = n_days // 7
    week_start = cutoff + 1 - 7 * (n_weeks - np.arange(n_weeks))
    starts = [int(s) for s in week_start]
    weekly_samples = _weekly(samples, n_weeks)
    point_weekly = _weekly(point, n_weeks)

    def label(day: int) -> object:
        if origin is None:
            return int(day)
        return (origin + dt.timedelta(days=int(day))).isoformat()

    day_labels = [label(s) for s in range(n_days)]
    return {
        "method": method,
        "cutoff": cutoff,
        "dropped": prep.dropped,
        "delay": f,
        "delay_by_period": f_periods,
        "period_start": period_start,
        "reported": reported,
        "point": point,
        "point_weekly": point_weekly,
        "week_start": week_start.astype(np.int64),
        "samples": samples,
        "weekly_samples": weekly_samples,
        "quantiles": levels,
        "daily_quantiles": _quantile_rows(samples, levels, day_labels),
        "weekly_quantiles": _quantile_rows(
            weekly_samples,
            levels,
            [label(s + 6) for s in starts],
            [label(s) for s in starts],
        ),
    }


def write_nowcast_csv(
    result: NowcastResult,
    path: str | os.PathLike[str],
    by: str = "week",
) -> None:
    """Write the quantile table of a nowcast as CSV.

    The table is in long format with one row per date and quantile, the
    format forecast hubs use for quantile forecasts.
    Columns are `target_end_date`, `quantile` and `value`, with
    `week_start` first for weekly tables.

    Parameters
    ----------
    result : NowcastResult
        Output of `nowcast_linelist`.
    path : str or os.PathLike
        File to write.
    by : {"week", "day"}, optional
        Write the weekly or the daily table.

    Raises
    ------
    ValueError
        If `by` is not `week` or `day`.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> import numpy as np
    >>> onset = np.array([0, 1, np.nan, 3, 4, np.nan, 6, 8, np.nan, 9])
    >>> report = np.array([2, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> res = nowcast_linelist(onset, report, n_samples=50)
    >>> out = pathlib.Path(tempfile.mkdtemp()) / "nowcast.csv"
    >>> write_nowcast_csv(res, out)
    >>> out.read_text().splitlines()[0]
    'week_start,target_end_date,quantile,value'
    """
    if by == "week":
        rows = result["weekly_quantiles"]
        fields = ["week_start", "target_end_date", "quantile", "value"]
    elif by == "day":
        rows = result["daily_quantiles"]
        fields = ["target_end_date", "quantile", "value"]
    else:
        raise ValueError(f"by must be 'week' or 'day', got {by!r}")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
