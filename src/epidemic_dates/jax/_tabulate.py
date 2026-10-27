# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Tabulate a line list into the arrays `linelist_loglik` needs."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..delays import DMAX

__all__ = ["LinelistArrays", "tabulate_linelist"]


class LinelistArrays(NamedTuple):
    """Line list counts for `linelist_loglik`.

    Attributes
    ----------
    complete : numpy.ndarray of float
        Complete-case counts by onset day (rows, 0..cutoff) and delay
        (columns, 0..dmax).
    missing_by_report : numpy.ndarray of float
        Missing-onset counts by report day 0..cutoff.
    cutoff : int
        Data cut-off day T.
    dropped : int
        Number of cases dropped.
    """

    complete: NDArray[np.float64]
    missing_by_report: NDArray[np.float64]
    cutoff: int
    dropped: int


def tabulate_linelist(
    onset: ArrayLike,
    report: ArrayLike,
    cutoff: int | None = None,
    dmax: int = DMAX,
) -> LinelistArrays:
    """Count a line list by onset day and delay, and by report day.

    Runs in NumPy, outside `jit`, since the output shapes depend on
    `cutoff` and `dmax`.

    Parameters
    ----------
    onset : array_like
        Onset day per case, with NaN where missing.
    report : array_like of int
        Report day per case. Must have no missing values.
    cutoff : int, optional
        Data cut-off day T. Defaults to the last report day. Cases
        reported before day 0 or after T are dropped.
    dmax : int, optional
        Maximum delay. Defaults to `DMAX`.

    Returns
    -------
    LinelistArrays
        Complete-case counts of shape (T + 1, dmax + 1), missing-onset
        counts by report day of length T + 1, the cut-off and the number
        of cases dropped. Complete cases with a negative onset, an onset
        after report or a delay over `dmax` are dropped.

    Raises
    ------
    ValueError
        If `onset` and `report` differ in length or are not
        one-dimensional, a report day is missing, a day is not a whole
        number, or `cutoff` or `dmax` is negative.

    Examples
    --------
    >>> import numpy as np
    >>> arrays = tabulate_linelist([0, np.nan, 2], [3, 4, 5], dmax=10)
    >>> arrays.complete.shape
    (6, 11)
    >>> arrays.missing_by_report.tolist()
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    """
    onset_days = np.asarray(onset, dtype=np.float64)
    report_days = np.asarray(report, dtype=np.float64)
    if onset_days.ndim != 1 or onset_days.shape != report_days.shape:
        raise ValueError(
            "onset and report must be one-dimensional and the same length"
        )
    if not np.all(np.isfinite(report_days)):
        raise ValueError("every case needs a finite report day")
    if np.isinf(onset_days).any():
        raise ValueError("onset days must not be infinite; use NaN if missing")
    for values, what in ((report_days, "report"), (onset_days, "onset")):
        finite = values[np.isfinite(values)]
        if not np.all(finite == np.round(finite)):
            raise ValueError(f"{what} days must be whole numbers")
    if dmax < 0:
        raise ValueError(f"dmax must be non-negative, got {dmax}")
    if cutoff is None:
        cutoff = int(np.max(report_days)) if report_days.size else 0
    if cutoff < 0 or cutoff != int(cutoff):
        raise ValueError(f"cutoff must be a non-negative integer, got {cutoff}")
    cutoff = int(cutoff)
    n_days = cutoff + 1

    report_int = report_days.astype(np.int64)
    keep = (report_int >= 0) & (report_int <= cutoff)
    onset_days, report_int = onset_days[keep], report_int[keep]
    missing = np.isnan(onset_days)
    onset_c = onset_days[~missing].astype(np.int64)
    delay_c = report_int[~missing] - onset_c
    ok = (onset_c >= 0) & (delay_c >= 0) & (delay_c <= dmax)

    complete = np.zeros((n_days, dmax + 1))
    np.add.at(complete, (onset_c[ok], delay_c[ok]), 1.0)
    missing_by_report = np.bincount(
        report_int[missing], minlength=n_days
    ).astype(np.float64)
    return LinelistArrays(
        complete,
        missing_by_report,
        cutoff,
        int((~keep).sum() + (~ok).sum()),
    )
