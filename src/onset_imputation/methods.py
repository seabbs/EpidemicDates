# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Imputation methods for cases with a missing onset.

Complete cases keep their true onset.
Cases with a missing onset are imputed by one of `METHODS`:

- `hub`: onset = report - delay drawn from the pooled empirical delays
  of complete cases reported by the cut-off (`hub_allocate` gives the
  expected allocation).
- `mean`: onset = report - round(mean empirical delay), with halves
  rounded up (`mean_shift_allocate`).
- `forward`: onset = report - delay drawn from the true forward delay
  pmf. An oracle, available in simulation only.
- `em`: EM back-projection (`em_backproject`). $P(s \mid t)$ is
  proportional to $\lambda(s) f(t - s)$, with onsets $\lambda$ and a
  static delay $f$ estimated jointly from all cases with right
  truncation at the cut-off.
"""

from __future__ import annotations

import math
from typing import Final, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .delays import DMAX

__all__ = [
    "METHODS",
    "EMFit",
    "em_backproject",
    "em_fit",
    "hub_allocate",
    "mean_shift_allocate",
    "round_half_up",
]

METHODS: Final[tuple[str, ...]] = ("hub", "mean", "forward", "em")
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


def hub_allocate(
    missing_by_report: NDArray[np.float64], delay_pmf: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Allocate missing-onset cases back from report day by a delay pmf.

    This is the expected allocation of the `hub` method: each case
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
        Estimated static forward delay pmf $f(d)$ on 0..DMAX.
    """

    allocation: NDArray[np.float64]
    onsets: NDArray[np.float64]
    delay: NDArray[np.float64]


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
) -> EMFit:
    r"""Back-project missing-onset cases by EM, returning all estimates.

    Expected counts by onset day $s$ and delay $d$ are
    $\lambda(s) f(d)$, observed only when $s + d \le T$ (right
    truncation), where $T$ is `cutoff`.
    Complete cases give counts $n_c(s, d)$.
    Missing cases give counts $m(t)$ by report day only.

    Starting values are the complete-case delay counts plus 0.5,
    normalised, for $f$, and complete-case onset counts inflated by the
    overall missing fraction, plus 0.1, for $\lambda$.
    Each of `n_iter` iterations then does the following.

    E-step: allocate the missing cases reported on day $t$ as

    $$n_a(s, t - s) = m(t) \frac{\lambda(s) f(t - s)}
    {\sum_{s'} \lambda(s') f(t - s')}.$$

    M-step: with $n = n_c + n_a$, alternate three times between

    $$\lambda(s) = \frac{\sum_d n(s, d)}{F\big(\min(T - s, D)\big)},
    \qquad
    f(d) \propto \frac{\sum_s n(s, d)}{\sum_{s \le T - d} \lambda(s)},$$

    where $F$ is the delay CDF and $D$ = `DMAX`.
    $f$ is floored at $10^{-12}$ and renormalised.

    The delay is static over time.
    The method assumes onsets are missing at random given the report
    day.
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

    Returns
    -------
    EMFit
        The expected allocation of missing-onset cases by onset day
        (length `cutoff + 1`, summing to `len(missing_report)`), the
        estimated onset curve and the estimated delay pmf. The onset
        curve and delay are those after the last M-step; the allocation
        is from the last E-step.

    Raises
    ------
    ValueError
        If the inputs are out of range, not whole numbers, or of
        mismatched length, or if `n_iter` is below 1.
    """
    if cutoff < 0:
        raise ValueError(f"cutoff must be non-negative, got {cutoff}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}")
    cutoff = int(cutoff)
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

    n_days = cutoff + 1
    n_complete = np.zeros((n_days, DMAX + 1))
    np.add.at(n_complete, (onset_c, delay_c), 1.0)
    missing = np.bincount(report_m, minlength=n_days)[:n_days].astype(
        np.float64
    )
    onset_grid, delay_grid = np.meshgrid(
        np.arange(n_days), np.arange(DMAX + 1), indexing="ij"
    )
    report_grid = onset_grid + delay_grid
    observed = report_grid <= cutoff
    report_idx = np.where(observed, report_grid, 0)
    delay = n_complete.sum(0) + 0.5
    delay /= delay.sum()
    n_cases = missing.sum() + n_complete.sum()
    missing_frac = missing.sum() / n_cases if n_cases > 0 else 0.0
    onsets = n_complete.sum(1) / max(1e-9, 1 - missing_frac) + 0.1
    allocated = np.zeros_like(n_complete)
    days_left = np.minimum(cutoff - np.arange(n_days), DMAX)
    delays = np.arange(DMAX + 1)
    for _ in range(n_iter):
        expected = onsets[:, None] * delay[None, :] * observed
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
        for _ in range(3):
            delay_cdf = np.cumsum(delay)
            onsets = total.sum(1) / delay_cdf[days_left]
            cum_onsets = np.cumsum(onsets)
            exposure = np.where(
                cutoff - delays >= 0,
                cum_onsets[np.maximum(cutoff - delays, 0)],
                0.0,
            )
            delay = total.sum(0) / np.maximum(exposure, 1e-300)
            delay = np.maximum(delay, 1e-12)
            delay /= delay.sum()
    return EMFit(allocated.sum(1), onsets, delay)


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
