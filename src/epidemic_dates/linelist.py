# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Impute missing onsets in a real line list."""

from __future__ import annotations

from typing import NamedTuple, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .delays import DMAX
from .methods import (
    DELAY_SHRINKAGE,
    as_delay,
    backsampling_allocate,
    em_fit,
    mean_shift_allocate,
    posterior_allocate,
    round_half_up,
)

__all__ = ["LinelistResult", "impute_linelist"]


class LinelistResult(TypedDict):
    """Output of `impute_linelist`.

    Daily arrays have length T + 1 and index i is onset day i.

    Attributes
    ----------
    complete : numpy.ndarray of float
        Onset counts of cases with a known onset.
    missing_by_report : numpy.ndarray of float
        Counts of missing-onset cases by report day.
    empirical_delay : numpy.ndarray of float
        Pooled delay pmf (0..DMAX) of complete cases.
    em_delay : numpy.ndarray of float
        Delay pmf (0..DMAX) estimated by EM back-projection: the static
        delay, or with `delay_period` the pooled delay.
    em_delay_by_period : numpy.ndarray of float
        Delay pmf of each period, shape `(n_periods, DMAX + 1)`. One row,
        equal to `em_delay`, without `delay_period`.
    delay_period_start : numpy.ndarray of int
        First onset day of each period. `[0]` without `delay_period`.
    imputed : dict of str to numpy.ndarray
        Daily onset counts (complete plus imputed) for `backsampling`,
        `mean` and `em`, and `forward` if a delay is given.
    imputed_before_day0 : dict of str to float
        Expected number of missing-onset cases that `backsampling`, `mean`
        and `forward` place before day 0. These are not in `imputed`.
    dropped : int
        Number of cases dropped before imputation.
    """

    complete: NDArray[np.float64]
    missing_by_report: NDArray[np.float64]
    empirical_delay: NDArray[np.float64]
    em_delay: NDArray[np.float64]
    em_delay_by_period: NDArray[np.float64]
    delay_period_start: NDArray[np.int64]
    imputed: dict[str, NDArray[np.float64]]
    imputed_before_day0: dict[str, float]
    dropped: int


def _check_whole(values: NDArray[np.float64], what: str) -> None:
    """Raise if any finite value is not a whole number."""
    finite = values[np.isfinite(values)]
    if not np.all(finite == np.round(finite)):
        raise ValueError(f"{what} must be whole numbers of days")


class _Prepared(NamedTuple):
    """A validated line list, split into complete and missing cases."""

    onset: NDArray[np.int64]
    delay: NDArray[np.int64]
    missing_report: NDArray[np.int64]
    cutoff: int
    dropped: int


def _prepare(
    onset: ArrayLike, report: ArrayLike, cutoff: int | None
) -> _Prepared:
    """Validate a line list and drop cases that cannot be used.

    Shared by `impute_linelist` and `nowcast_linelist`. See
    `impute_linelist` for the checks and the cases dropped.
    """
    onset_days = np.asarray(onset, dtype=np.float64)
    report_days = np.asarray(report, dtype=np.float64)
    if onset_days.ndim != 1 or report_days.ndim != 1:
        raise ValueError("onset and report must be one-dimensional")
    if onset_days.size != report_days.size:
        raise ValueError(
            "onset and report must have the same length, got "
            f"{onset_days.size} and {report_days.size}"
        )
    if report_days.size == 0:
        raise ValueError("the line list is empty")
    bad_report = ~np.isfinite(report_days)
    if bad_report.any():
        raise ValueError(
            f"report day is missing or infinite for {int(bad_report.sum())} "
            "case(s); every case needs a report day"
        )
    if np.isinf(onset_days).any():
        raise ValueError("onset days must not be infinite; use NaN if missing")
    _check_whole(report_days, "report days")
    _check_whole(onset_days, "onset days")
    if cutoff is None:
        cutoff = int(np.max(report_days))
    elif cutoff != int(cutoff):
        raise ValueError(f"cutoff must be a whole number, got {cutoff}")
    cutoff = int(cutoff)
    if cutoff < 0:
        raise ValueError(f"cutoff must be non-negative, got {cutoff}")

    report_int = report_days.astype(np.int64)
    keep = (report_int >= 0) & (report_int <= cutoff)
    onset_days, report_int = onset_days[keep], report_int[keep]
    missing = np.isnan(onset_days)
    onset_c = onset_days[~missing].astype(np.int64)
    delay_c = report_int[~missing] - onset_c
    ok = (onset_c >= 0) & (delay_c >= 0) & (delay_c <= DMAX)
    onset_c, delay_c = onset_c[ok], delay_c[ok]
    if onset_c.size == 0:
        raise ValueError(
            "no complete cases (cases with both an onset and a report day "
            "in range) are left, so the delay cannot be estimated"
        )
    return _Prepared(
        onset_c,
        delay_c,
        report_int[missing],
        cutoff,
        int((~keep).sum() + (~ok).sum()),
    )


def impute_linelist(
    onset: ArrayLike,
    report: ArrayLike,
    cutoff: int | None = None,
    n_iter: int = 150,
    delay: ArrayLike | None = None,
    delay_period: int | None = None,
    delay_shrinkage: float = DELAY_SHRINKAGE,
) -> LinelistResult:
    r"""Impute missing onsets in a real line list.

    Parameters
    ----------
    onset : array_like
        Onset day per case, with NaN (or None) where missing.
    report : array_like of int
        Report day per case. Must have no missing values. Days are
        integers, for example days since the first report.
    cutoff : int, optional
        Data cut-off day T. Must be non-negative. Defaults to the last
        report day. Cases reported before day 0 or after T are dropped.
    n_iter : int, optional
        Number of EM iterations passed to `em_fit`. Must be at least 1.
    delay : array_like of float, optional
        A known forward delay, used as given (a plug-in). Either one pmf
        indexed by delay in days, or a matrix with one pmf per onset day
        0..T. If given, `em` holds it fixed instead of estimating a
        delay, and `imputed` also holds `forward`. Cannot be combined
        with `delay_period`.
    delay_period : int, optional
        Length in days of the periods of onset days that get their own
        EM delay, counted back from the cut-off. `None` (the default)
        estimates one static delay. See `em_fit`.
    delay_shrinkage : float, optional
        Strength, in expected onsets, of the shrinkage of each period's
        delay towards the previous period's delay. Must be positive.
        Only used with `delay_period`.

    Returns
    -------
    LinelistResult
        Daily arrays have length T + 1 and index i is onset day i.

        - `complete`: onset counts of cases with a known onset.
        - `missing_by_report`: counts of missing-onset cases by report
          day.
        - `empirical_delay`: pooled delay pmf (0..DMAX) of complete
          cases.
        - `em_delay`: delay pmf (0..DMAX) used by EM: the static
          estimate, the pooled estimate with `delay_period`, or the given
          `delay` (its onset-weighted mean if it is a matrix).
        - `em_delay_by_period`: delay pmf of each period, one row per
          period.
        - `delay_period_start`: first onset day of each period.
        - `imputed`: dict of daily onset counts (complete plus imputed)
          for `backsampling`, `mean` and `em`, and `forward` with
          `delay`.
        - `imputed_before_day0`: dict with the expected number of
          missing-onset cases that `backsampling`, `mean` and `forward`
          place before day 0.
        - `dropped`: number of cases dropped.

    Raises
    ------
    ValueError
        If the line list is empty, `onset` and `report` differ in
        length, a report day is missing or infinite, a day is not a
        whole number, `cutoff` is negative, no complete case is left
        after dropping, `delay` is not a valid pmf or matrix of pmfs,
        `delay_period` or `delay_shrinkage` is out of range, or both
        `delay` and `delay_period` are given.

    Notes
    -----
    - `backsampling` spreads each missing case over report day minus each delay,
      weighted by `empirical_delay`. It is the expected allocation of
      the random draw, not a draw.
    - `mean` shifts each missing case back by the mean of
      `empirical_delay`, rounded with halves up (as JavaScript's
      `Math.round`).
    - `em` uses `em_fit`. It estimates a static delay jointly with the
      onset curve, allowing for right truncation at the cut-off. With
      `delay_period` it estimates one delay per period instead, each
      shrunk towards the previous period's delay. With `delay` it uses
      that delay as given and estimates only the onset curve. `backsampling` and
      `mean` always use the pooled empirical delay of the data.
    - `forward` places each missing case by the given delay,
      $P(s \mid t) = f_s(t - s)$. With a static delay mass before day 0
      is lost, as for `backsampling`. With a matrix the weights are
      renormalised over onset days 0..t.
    - All methods assume onsets are missing at random given the report
      day. If missingness depends on the delay, all are biased.
    - Complete cases with a negative onset, an onset after report or a
      delay over `DMAX` are dropped. They are not imputed.
    - For `backsampling` and `mean`, imputed onsets before day 0 are not in
      `imputed`, so totals can be below the number of cases kept. The
      amount is in `imputed_before_day0`.
    - Outputs are expected counts and need not be whole numbers.
      Sum to weeks before comparing methods.

    Examples
    --------
    >>> import numpy as np
    >>> onset = np.array([0, 2, np.nan, 5, np.nan])
    >>> report = np.array([3, 6, 7, 9, 12])
    >>> res = impute_linelist(onset, report, cutoff=12)
    >>> res["imputed"]["backsampling"].shape
    (13,)
    """
    prep = _prepare(onset, report, cutoff)
    onset_c, delay_c, report_m = prep.onset, prep.delay, prep.missing_report
    cutoff = prep.cutoff
    n_days = cutoff + 1
    fixed = None if delay is None else as_delay(delay, cutoff)
    complete = np.bincount(onset_c, minlength=n_days)[:n_days].astype(
        np.float64
    )
    missing_by_report = np.bincount(report_m, minlength=n_days)[:n_days].astype(
        np.float64
    )
    empirical = np.bincount(delay_c, minlength=DMAX + 1).astype(np.float64)
    empirical /= empirical.sum()

    backsampled = backsampling_allocate(missing_by_report, empirical)
    shift = round_half_up(float(np.dot(np.arange(DMAX + 1), empirical)))
    mean = mean_shift_allocate(missing_by_report, shift)
    fit = em_fit(
        (onset_c, delay_c),
        report_m,
        cutoff,
        n_iter=n_iter,
        delay=fixed,
        delay_period=delay_period,
        delay_shrinkage=delay_shrinkage,
    )
    n_missing = float(missing_by_report.sum())
    imputed = {
        "backsampling": complete + backsampled,
        "mean": complete + mean,
        "em": complete + fit.allocation,
    }
    before_day0 = {
        "backsampling": max(n_missing - float(backsampled.sum()), 0.0),
        "mean": n_missing - float(mean.sum()),
    }
    if fixed is not None:
        if fixed.ndim == 1:
            forward = backsampling_allocate(missing_by_report, fixed)
        else:
            forward = posterior_allocate(
                missing_by_report, np.ones(n_days), fixed
            )
        imputed["forward"] = complete + forward
        before_day0["forward"] = max(n_missing - float(forward.sum()), 0.0)
    return {
        "complete": complete,
        "missing_by_report": missing_by_report,
        "empirical_delay": empirical,
        "em_delay": fit.delay,
        "em_delay_by_period": fit.delay_by_period,
        "delay_period_start": fit.period_start,
        "imputed": imputed,
        "imputed_before_day0": before_day0,
        "dropped": prep.dropped,
    }
