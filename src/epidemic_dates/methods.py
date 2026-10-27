# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Imputation methods for cases with a missing onset.

Complete cases keep their true onset.
Cases with a missing onset are imputed by one of `METHODS`:

- `backsampling`: onset = report - delay drawn from the pooled empirical delays
  of complete cases reported by the cut-off (`backsampling_allocate` gives the
  expected allocation).
- `mean`: onset = report - round(mean empirical delay), with halves
  rounded up (`mean_shift_allocate`).
- `forward`: onset = report - delay drawn from the true forward delay
  pmf. An oracle, available in simulation only.
- `em`: EM back-projection (`em_backproject`). $P(s \mid t)$ is
  proportional to $\lambda(s) f(t - s)$, with onsets $\lambda$ and a
  static delay $f$ estimated jointly from all cases with right
  truncation at the cut-off. With `delay_period`, each period of onset
  days gets its own delay $f_k$, shrunk towards the previous period's.
"""

from __future__ import annotations

import math
from typing import Final, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .delays import DMAX

__all__ = [
    "DELAY_SHRINKAGE",
    "METHODS",
    "EMFit",
    "as_delay",
    "backsampling_allocate",
    "delay_periods",
    "em_backproject",
    "em_fit",
    "mean_shift_allocate",
    "posterior_allocate",
    "round_half_up",
]

METHODS: Final[tuple[str, ...]] = ("backsampling", "mean", "forward", "em")
"""Imputation methods compared, in the order used in summary rows."""


def round_half_up(x: float) -> int:
    """Round to the nearest integer, with halves rounded up.

    This matches JavaScript's `Math.round`. Python's built-in `round`
    rounds halves to even.

    Parameters
    ----------
    x : float
        Value to round. Must be finite.

    Returns
    -------
    int
        `floor(x + 0.5)`.

    Raises
    ------
    ValueError
        If `x` is not finite.

    Examples
    --------
    >>> round_half_up(2.5), round(2.5)
    (3, 2)
    """
    if not math.isfinite(x):
        raise ValueError(f"cannot round non-finite value {x}")
    return math.floor(x + 0.5)


def backsampling_allocate(
    missing_by_report: NDArray[np.float64], delay_pmf: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Allocate missing-onset cases back from report day by a delay pmf.

    This is the expected allocation of the `backsampling` method: each case
    reported on day t is spread over onset days t - d with weight
    `delay_pmf[d]`. It is not a random draw.

    Parameters
    ----------
    missing_by_report : numpy.ndarray of float
        Counts of missing-onset cases by report day 0..T.
    delay_pmf : numpy.ndarray of float
        Delay pmf on 0..D.

    Returns
    -------
    numpy.ndarray of float
        Expected missing-onset cases by onset day 0..T, the same length
        as `missing_by_report`. Mass that falls before day 0 is not
        included, so the sum can be below `missing_by_report.sum()`.
    """
    n_days = missing_by_report.size
    allocation = np.zeros(n_days)
    for delay in range(min(delay_pmf.size, n_days)):
        allocation[: n_days - delay] += (
            missing_by_report[delay:] * delay_pmf[delay]
        )
    return allocation


def mean_shift_allocate(
    missing_by_report: NDArray[np.float64], shift: int
) -> NDArray[np.float64]:
    """Move missing-onset cases back from report day by a fixed shift.

    This is the `mean` method.

    Parameters
    ----------
    missing_by_report : numpy.ndarray of float
        Counts of missing-onset cases by report day 0..T.
    shift : int
        Days to move each case back. Must be non-negative.

    Returns
    -------
    numpy.ndarray of float
        Missing-onset cases by onset day 0..T, the same length as
        `missing_by_report`. Cases moved before day 0 are not included.

    Raises
    ------
    ValueError
        If `shift` is negative.
    """
    if shift < 0:
        raise ValueError(f"shift must be non-negative, got {shift}")
    n_days = missing_by_report.size
    allocation = np.zeros(n_days)
    if shift < n_days:
        allocation[: n_days - shift] = missing_by_report[shift:]
    return allocation


def posterior_allocate(
    missing_by_report: NDArray[np.float64],
    onsets: NDArray[np.float64],
    delay_by_day: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Allocate missing-onset cases by $\lambda(s) f_s(t - s)$.

    Each case reported on day $t$ is spread over onset days
    $s = 0, \dots, t$ with weight proportional to
    $\lambda(s) f_s(t - s)$, where $f_s$ is row $s$ of `delay_by_day`.
    With $\lambda$ constant this is the forward draw with a delay that
    changes over time.
    Weights are renormalised over $0, \dots, t$, so no case is placed
    before day 0.
    Cases on a report day with no possible onset are dropped.

    Parameters
    ----------
    missing_by_report : numpy.ndarray of float
        Counts of missing-onset cases by report day 0..T.
    onsets : numpy.ndarray of float
        Onset curve $\lambda$ on 0..T.
    delay_by_day : numpy.ndarray of float
        Delay pmf of each onset day, shape `(T + 1, n_delays)`.

    Returns
    -------
    numpy.ndarray of float
        Expected missing-onset cases by onset day 0..T.

    Examples
    --------
    >>> f = np.array([[0.5, 0.5], [1.0, 0.0]])
    >>> posterior_allocate(np.array([0.0, 3.0]), np.ones(2), f).tolist()
    [1.0, 2.0]
    """
    n_days = missing_by_report.size
    out = np.zeros(n_days)
    n_delays = delay_by_day.shape[1]
    for t in np.flatnonzero(missing_by_report):
        d = np.arange(min(n_delays, t + 1))
        w = onsets[t - d] * delay_by_day[t - d, d]
        if w.sum() <= 0:
            continue
        out[t - d] += missing_by_report[t] * w / w.sum()
    return out


class EMFit(NamedTuple):
    r"""Result of `em_fit`.

    Attributes
    ----------
    allocation : numpy.ndarray of float
        Expected number of missing-onset cases on each onset day 0..T.
    onsets : numpy.ndarray of float
        Estimated expected onsets $\lambda(s)$ of all cases, including
        those not yet reported, on each onset day 0..T.
    delay : numpy.ndarray of float
        Estimated pooled forward delay pmf $f(d)$ on 0..DMAX. With one
        period this is the static delay. With several it is the pooled
        delay, which the first period is shrunk towards.
    delay_by_period : numpy.ndarray of float
        Forward delay pmf $f_k(d)$ of each period, shape
        `(n_periods, DMAX + 1)`. One row, equal to `delay`, without
        periods.
    period_start : numpy.ndarray of int
        First onset day of each period. `[0]` without periods.
    """

    allocation: NDArray[np.float64]
    onsets: NDArray[np.float64]
    delay: NDArray[np.float64]
    delay_by_period: NDArray[np.float64]
    period_start: NDArray[np.int64]


DELAY_SHRINKAGE: Final[float] = 10.0
"""Default strength of the shrinkage of each period's delay towards the
previous period's delay, in expected onsets. See `em_fit`."""


def delay_periods(cutoff: int, delay_period: int | None) -> NDArray[np.int64]:
    """Split onset days 0..cutoff into periods that end on the cut-off.

    Periods are `delay_period` days long and are counted back from the
    cut-off, so the latest period is always full length.
    The first period holds the days left over and can be shorter.

    Parameters
    ----------
    cutoff : int
        Data cut-off day T. Must be non-negative.
    delay_period : int or None
        Period length in days, at least 1. `None` gives one period.

    Returns
    -------
    numpy.ndarray of int
        Period index (0, 1, ...) of each onset day 0..cutoff.

    Raises
    ------
    ValueError
        If `delay_period` is not a whole number of at least 1.

    Examples
    --------
    >>> delay_periods(9, 4).tolist()
    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    """
    days = np.arange(cutoff + 1)
    if delay_period is None:
        return np.zeros(cutoff + 1, dtype=np.int64)
    if (
        isinstance(delay_period, bool)
        or not float(delay_period).is_integer()
        or delay_period < 1
    ):
        raise ValueError(
            "delay_period must be a whole number of days of at least 1, "
            f"got {delay_period}"
        )
    period = int(delay_period)
    n_periods = -(-(cutoff + 1) // period)
    return (n_periods - 1 - (cutoff - days) // period).astype(np.int64)


def _check_shrinkage(delay_shrinkage: float) -> float:
    """Return `delay_shrinkage` as a float, or raise if not positive."""
    value = float(delay_shrinkage)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(
            "delay_shrinkage must be a positive finite number, "
            f"got {delay_shrinkage}"
        )
    return value


def _as_int_days(values: ArrayLike, what: str) -> NDArray[np.int64]:
    """Convert to a 1-D integer array, rejecting non-integer values."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        as_float = arr.astype(np.float64)
        if not np.all(np.isfinite(as_float)):
            raise ValueError(f"{what} must not contain NaN or infinite values")
        if not np.all(as_float == np.round(as_float)):
            raise ValueError(f"{what} must be whole numbers of days")
    return arr.astype(np.int64)


def em_fit(
    complete: tuple[ArrayLike, ArrayLike],
    missing_report: ArrayLike,
    cutoff: int,
    n_iter: int = 150,
    delay: ArrayLike | None = None,
    delay_period: int | None = None,
    delay_shrinkage: float = DELAY_SHRINKAGE,
) -> EMFit:
    r"""Back-project missing-onset cases by EM, returning all estimates.

    Onset days are split into periods $k$ of `delay_period` days, counted
    back from the cut-off (see `delay_periods`).
    By default there is one period, so the delay is static.
    Expected counts by onset day $s$ and delay $d$ are
    $\lambda(s) f_{k(s)}(d)$, observed only when $s + d \le T$ (right
    truncation), where $T$ is `cutoff` and $k(s)$ is the period of onset
    day $s$.
    Complete cases give counts $n_c(s, d)$.
    Missing cases give counts $m(t)$ by report day only.

    Starting values are the complete-case delay counts plus 0.5,
    normalised, for every $f_k$, and complete-case onset counts inflated
    by the overall missing fraction, plus 0.1, for $\lambda$.
    Each of `n_iter` iterations then does the following.

    E-step: allocate the missing cases reported on day $t$ as

    $$n_a(s, t - s) = m(t) \frac{\lambda(s) f_{k(s)}(t - s)}
    {\sum_{s'} \lambda(s') f_{k(s')}(t - s')}.$$

    M-step: with $n = n_c + n_a$, alternate three times between

    $$\lambda(s) = \frac{\sum_d n(s, d)}{F_{k(s)}\big(\min(T - s, D)\big)},
    \qquad
    f(d) \propto \frac{\sum_s n(s, d)}{\sum_{s \le T - d} \lambda(s)},$$

    where $F_k$ is the CDF of $f_k$ and $D$ = `DMAX`.
    $f$ is the pooled delay, floored at $10^{-12}$ and renormalised.
    With one period $f_0 = f$.
    With several, each period's delay is shrunk towards the delay of the
    period before it, and the first period towards $f$.
    In order $k = 0, 1, \dots$,

    $$f_k(d) \propto \frac{N_k(d) + \kappa h_k(d)}{E_k(d) + \kappa},
    \qquad
    h_0 = f, \quad h_k = f_{k-1},
    \qquad
    N_k(d) = \sum_{s \in k} n(s, d),
    \qquad
    E_k(d) = \sum_{s \in k,\, s \le T - d} \lambda(s),$$

    with $\kappa$ = `delay_shrinkage`.
    $E_k(d)$ is the expected number of period-$k$ onsets that could be
    seen with delay $d$ by the cut-off, so the weight on the period's own
    data, $E_k(d) / (E_k(d) + \kappa)$, grows with its effective sample
    size.
    It is the posterior mode of $f_k(d)$ under a
    $\operatorname{Gamma}\big(\kappa h_k(d) + 1, \kappa\big)$ prior,
    whose mode is $h_k(d)$, so the periods follow a random walk.
    A delay a period cannot yet show ($E_k(d) = 0$), and every delay of a
    period with no cases, takes the previous period's value.
    This matters most for the latest period, which can only show delays
    up to its length: its longer delays come from the period before.
    $f_k$ is floored at $10^{-12}$ and renormalised.

    If `delay` is given it is a plug-in: it is held fixed and only
    $\lambda$ is updated in the M-step.
    A one-dimensional `delay` is a static delay.
    A two-dimensional `delay` has one row per onset day, so the delay of
    onset day $s$ is row $s$ and can change over time.

    The method assumes onsets are missing at random given the report
    day, and that the delay is constant within each period.
    No smoothing is applied to $\lambda$.

    Parameters
    ----------
    complete : tuple of array_like
        `(onset_days, delays)`: integer onset days and delays of complete
        cases. Onset days must lie in 0..cutoff and delays in 0..DMAX,
        with onset plus delay <= cutoff.
    missing_report : array_like of int
        Report days (0..cutoff) of cases with a missing onset.
    cutoff : int
        Data cut-off day T. Must be non-negative.
    n_iter : int, optional
        Number of EM iterations. Must be at least 1. No convergence
        check is made.
    delay : array_like of float, optional
        Forward delay to hold fixed instead of estimating it. Either one
        pmf indexed by delay in days, or a matrix with one such pmf per
        onset day 0..cutoff (shape `(cutoff + 1, n_delays)`). Entries
        beyond `DMAX` are dropped, missing entries up to `DMAX` are
        taken as 0 and each pmf is renormalised. Values must be finite
        and non-negative with a positive sum. Cannot be combined with
        `delay_period`.
    delay_period : int, optional
        Length in days of the periods with their own delay. `None` (the
        default) estimates one static delay. A period at least
        `cutoff + 1` days long gives the same result as `None`.
    delay_shrinkage : float, optional
        Shrinkage strength $\kappa$, in expected onsets, of each
        period's delay towards the previous period's delay. Must be
        positive. Only
        used with `delay_period`.

    Returns
    -------
    EMFit
        The expected allocation of missing-onset cases by onset day
        (length `cutoff + 1`, summing to `len(missing_report)`), the
        estimated onset curve, the pooled delay pmf, the delay pmf of
        each period and the first day of each period. With a
        two-dimensional `delay`, each onset day is its own period and
        `delay` is the onset-weighted mean of its rows. The onset curve
        and delays are those after the last M-step; the allocation is
        from the last E-step.

    Raises
    ------
    ValueError
        If the inputs are out of range, not whole numbers, or of
        mismatched length, if `n_iter` is below 1, if `delay` is not
        a valid pmf, if `delay_period` or `delay_shrinkage` is out of
        range, or if both `delay` and `delay_period` are given.
    """
    if cutoff < 0:
        raise ValueError(f"cutoff must be non-negative, got {cutoff}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}")
    cutoff = int(cutoff)
    period = delay_periods(cutoff, delay_period)
    shrinkage = _check_shrinkage(delay_shrinkage)
    if delay is not None and delay_period is not None:
        raise ValueError("give either delay or delay_period, not both")
    onset_c = _as_int_days(complete[0], "complete onset days")
    delay_c = _as_int_days(complete[1], "complete delays")
    report_m = _as_int_days(missing_report, "missing_report")
    if onset_c.size != delay_c.size:
        raise ValueError(
            "complete onset days and delays must have the same length"
        )
    if np.any((onset_c < 0) | (onset_c > cutoff)):
        raise ValueError("complete onset days must lie in 0..cutoff")
    if np.any((delay_c < 0) | (delay_c > DMAX)):
        raise ValueError(f"complete delays must lie in 0..{DMAX}")
    if np.any(onset_c + delay_c > cutoff):
        raise ValueError("complete cases must be reported by the cutoff")
    if np.any((report_m < 0) | (report_m > cutoff)):
        raise ValueError("missing_report days must lie in 0..cutoff")

    fixed = None if delay is None else as_delay(delay, cutoff)
    if fixed is not None and fixed.ndim == 2:
        period = np.arange(cutoff + 1, dtype=np.int64)
    n_days = cutoff + 1
    n_complete = np.zeros((n_days, DMAX + 1))
    np.add.at(n_complete, (onset_c, delay_c), 1.0)
    missing = np.bincount(report_m, minlength=n_days)[:n_days].astype(
        np.float64
    )
    if fixed is None:
        start = n_complete.sum(0) + 0.5
        start /= start.sum()
    else:
        start = fixed
    n_cases = missing.sum() + n_complete.sum()
    missing_frac = missing.sum() / n_cases if n_cases > 0 else 0.0
    onsets = n_complete.sum(1) / max(1e-9, 1 - missing_frac) + 0.1
    return _em_iterate(
        n_complete,
        missing,
        cutoff,
        onsets,
        start,
        n_iter,
        fixed is not None,
        period,
        shrinkage,
    )


def as_delay(delay: ArrayLike, cutoff: int) -> NDArray[np.float64]:
    """Check a supplied delay, static or one pmf per onset day.

    Parameters
    ----------
    delay : array_like of float
        One delay pmf indexed by delay in days, or a matrix with one pmf
        per onset day 0..cutoff in its rows.
    cutoff : int
        Data cut-off day T.

    Returns
    -------
    numpy.ndarray of float
        The pmf on 0..DMAX, or a `(cutoff + 1, DMAX + 1)` matrix, with
        each pmf renormalised.

    Raises
    ------
    ValueError
        If `delay` is not one- or two-dimensional, a matrix does not have
        `cutoff + 1` rows, or a pmf is not valid (see `_as_delay_pmf`).

    Examples
    --------
    >>> as_delay([[1, 1], [0, 1]], cutoff=1).shape
    (2, 61)
    """
    arr = np.asarray(delay, dtype=np.float64)
    if arr.ndim == 1:
        return _as_delay_pmf(arr)
    if arr.ndim != 2:
        raise ValueError("delay must be a pmf or a matrix of pmfs")
    if arr.shape[0] != cutoff + 1:
        raise ValueError(
            "a delay matrix needs one row per onset day 0..cutoff "
            f"({cutoff + 1} rows), got {arr.shape[0]}"
        )
    return np.array([_as_delay_pmf(row) for row in arr])


def _as_delay_pmf(delay: ArrayLike) -> NDArray[np.float64]:
    """Check a delay pmf and return it on 0..DMAX, renormalised.

    Entries beyond `DMAX` are dropped and missing entries are taken as 0.

    Parameters
    ----------
    delay : array_like of float
        Delay pmf indexed by delay in days.

    Returns
    -------
    numpy.ndarray of float
        The pmf on 0..DMAX, summing to 1.

    Raises
    ------
    ValueError
        If `delay` is not one-dimensional, has a non-finite or negative
        value, or has no mass on 0..DMAX.
    """
    arr = np.asarray(delay, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("delay must be one-dimensional")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("delay must be finite and non-negative")
    out = np.zeros(DMAX + 1)
    n = min(arr.size, DMAX + 1)
    out[:n] = arr[:n]
    total = float(out.sum())
    if total <= 0:
        raise ValueError(f"delay must have positive mass on 0..{DMAX}")
    out /= total
    return out


def _em_iterate(
    n_complete: NDArray[np.float64],
    missing: NDArray[np.float64],
    cutoff: int,
    onsets: NDArray[np.float64],
    delay: NDArray[np.float64],
    n_iter: int,
    fix_delay: bool = False,
    period: NDArray[np.int64] | None = None,
    shrinkage: float = DELAY_SHRINKAGE,
) -> EMFit:
    r"""Run EM iterations from given starting values.

    The engine behind `em_fit`, exposed so a fit can be warm-started.
    See `em_fit` for the E-step and M-step.
    $F_k$ is floored at $10^{-12}$ before dividing.

    Parameters
    ----------
    n_complete : numpy.ndarray of float
        Complete-case counts by onset day (rows, 0..cutoff) and delay
        (columns, 0..DMAX).
    missing : numpy.ndarray of float
        Missing-onset cases by report day 0..cutoff.
    cutoff : int
        Data cut-off day T.
    onsets : numpy.ndarray of float
        Starting onset curve $\lambda$ on 0..cutoff.
    delay : numpy.ndarray of float
        Starting (or fixed) delay pmf on 0..DMAX, either one pmf used for
        every period or one row per period.
    n_iter : int
        Number of EM iterations.
    fix_delay : bool, optional
        Hold the delay fixed and update only $\lambda$.
    period : numpy.ndarray of int, optional
        Period index of each onset day, from `delay_periods`. One period
        if omitted.
    shrinkage : float, optional
        Shrinkage strength $\kappa$ of each period's delay.

    Returns
    -------
    EMFit
        As for `em_fit`.
    """
    n_days = cutoff + 1
    if period is None:
        period = np.zeros(n_days, dtype=np.int64)
    n_periods = int(period[-1]) + 1
    starts = np.flatnonzero(np.r_[True, np.diff(period) != 0]).astype(np.int64)
    onsets = np.array(onsets, dtype=np.float64)
    by_period = np.atleast_2d(np.array(delay, dtype=np.float64))
    if by_period.shape[0] != n_periods:
        by_period = np.tile(by_period[0], (n_periods, 1))
    pooled = by_period[0].copy()
    onset_grid, delay_grid = np.meshgrid(
        np.arange(n_days), np.arange(DMAX + 1), indexing="ij"
    )
    report_grid = onset_grid + delay_grid
    observed = report_grid <= cutoff
    report_idx = np.where(observed, report_grid, 0)
    allocated = np.zeros_like(n_complete)
    days_left = np.minimum(cutoff - np.arange(n_days), DMAX)
    delays = np.arange(DMAX + 1)
    last_seen = cutoff - delays
    # end day of each period and of the one before it
    ends = np.r_[starts[1:] - 1, cutoff]
    for _ in range(n_iter):
        expected = onsets[:, None] * by_period[period] * observed
        denom = np.bincount(
            report_idx.ravel(), weights=expected.ravel(), minlength=n_days
        )
        allocated = np.where(
            observed,
            expected * (missing / np.maximum(denom, 1e-300))[report_idx],
            0.0,
        )
        total = n_complete + allocated
        # M-step: right-truncated triangle (alternating updates)
        for _ in range(1 if fix_delay else 3):
            delay_cdf = np.maximum(np.cumsum(by_period, axis=1), 1e-12)
            onsets = total.sum(1) / delay_cdf[period, days_left]
            if fix_delay:
                continue
            cum_onsets = np.cumsum(onsets)
            exposure = np.where(
                last_seen >= 0,
                cum_onsets[np.maximum(last_seen, 0)],
                0.0,
            )
            pooled = total.sum(0) / np.maximum(exposure, 1e-300)
            pooled = np.maximum(pooled, 1e-12)
            pooled /= pooled.sum()
            if n_periods == 1:
                by_period = pooled[None, :]
                continue
            # per-period counts and exposure; shrink each period towards
            # the one before it, and the first towards the pooled delay
            counts = np.add.reduceat(total, starts, axis=0)
            before = np.r_[0.0, cum_onsets][starts]
            upto = np.minimum(last_seen[None, :], ends[:, None])
            seen = upto >= starts[:, None]
            period_exposure = np.where(
                seen, cum_onsets[np.maximum(upto, 0)] - before[:, None], 0.0
            )
            period_exposure = np.maximum(period_exposure, 0.0)
            target = pooled
            new = np.empty_like(counts)
            for k in range(n_periods):
                row = (counts[k] + shrinkage * target) / (
                    period_exposure[k] + shrinkage
                )
                row = np.maximum(row, 1e-12)
                new[k] = row / row.sum()
                target = new[k]
            by_period = new
    if fix_delay:
        if n_periods == 1:
            pooled = by_period[0].copy()
        else:
            weights = np.maximum(onsets, 0.0)
            if weights.sum() <= 0:
                weights = np.ones(n_days)
            pooled = weights @ by_period[period] / weights.sum()
    return EMFit(allocated.sum(1), onsets, pooled, by_period, starts)


def em_backproject(
    complete: tuple[ArrayLike, ArrayLike],
    missing_report: ArrayLike,
    cutoff: int,
    n_iter: int = 150,
) -> NDArray[np.float64]:
    """Back-project missing-onset cases by EM with an estimated delay.

    A shortcut for `em_fit(...).allocation`. See `em_fit` for the
    method.

    Parameters
    ----------
    complete : tuple of array_like
        `(onset_days, delays)`: integer onset days and delays of complete
        cases. Onset days must lie in 0..cutoff and delays in 0..DMAX,
        with onset plus delay <= cutoff.
    missing_report : array_like of int
        Report days (0..cutoff) of cases with a missing onset.
    cutoff : int
        Data cut-off day T. Must be non-negative.
    n_iter : int, optional
        Number of EM iterations. Must be at least 1. No convergence
        check is made.

    Returns
    -------
    numpy.ndarray of float
        Expected number of missing-onset cases on each onset day
        0..cutoff, length `cutoff + 1`. Sums to `len(missing_report)`.

    Raises
    ------
    ValueError
        If the inputs are invalid. See `em_fit`.
    """
    return em_fit(complete, missing_report, cutoff, n_iter=n_iter).allocation
