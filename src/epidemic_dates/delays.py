# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Onset-to-report delay distributions.

Delays are double interval censored daily Gamma pmfs on $0, \dots, D$
with $D$ = `DMAX`.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = ["DMAX", "F_DEFAULT", "gamma_pmf"]

DMAX: Final[int] = 60
"""Maximum onset-to-report delay in days. Delay pmfs cover 0..DMAX."""


def gamma_pmf(mean: float, sd: float, dmax: int = DMAX) -> NDArray[np.float64]:
    r"""Compute a double interval censored daily Gamma delay pmf.

    Uses the primarycensored form.
    The primary event (onset) is uniform within its day and the
    secondary event (report) is censored to its day, so

    $$P(d) = G(d + 1) - 2 G(d) + G(d - 1),$$

    $$G(x) = \int_0^x F(u; k, \theta) \, du
    = x F(x; k, \theta) - k \theta F(x; k + 1, \theta),$$

    where $F(x; k, \theta)$ is the Gamma CDF with shape
    $k = (\mu / \sigma)^2$ and scale $\theta = \sigma^2 / \mu$, and
    $G(x) = 0$ for $x \le 0$.
    The pmf is truncated at `dmax` and renormalised.

    Parameters
    ----------
    mean : float
        Mean of the continuous Gamma delay, in days. Must be positive.
    sd : float
        Standard deviation of the continuous Gamma delay, in days. Must
        be positive.
    dmax : int, optional
        Maximum delay. Must be non-negative. Defaults to `DMAX`.

    Returns
    -------
    numpy.ndarray of float
        Probabilities for delays 0..dmax, of length `dmax + 1`, summing
        to 1.

    Raises
    ------
    ValueError
        If `mean` or `sd` is not a positive finite number, or `dmax` is
        negative.

    Examples
    --------
    >>> pmf = gamma_pmf(8, 8)
    >>> pmf.shape
    (61,)
    >>> round(float(pmf.sum()), 12)
    1.0
    """
    if not (math.isfinite(mean) and mean > 0):
        raise ValueError(f"mean must be a positive finite number, got {mean}")
    if not (math.isfinite(sd) and sd > 0):
        raise ValueError(f"sd must be a positive finite number, got {sd}")
    if dmax < 0:
        raise ValueError(f"dmax must be non-negative, got {dmax}")
    shape = (mean / sd) ** 2
    scale = sd**2 / mean

    def integrated_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.maximum(x, 0.0)
        return np.asarray(
            x * stats.gamma.cdf(x, shape, scale=scale)
            - shape * scale * stats.gamma.cdf(x, shape + 1, scale=scale),
            dtype=np.float64,
        )

    days = np.arange(dmax + 1, dtype=np.float64)
    pmf = np.maximum(
        integrated_cdf(days + 1)
        - 2 * integrated_cdf(days)
        + integrated_cdf(days - 1),
        0.0,
    )
    return pmf / pmf.sum()


F_DEFAULT: Final[NDArray[np.float64]] = gamma_pmf(8, 8)
"""Default forward delay pmf, `gamma_pmf(8, 8)` on 0..DMAX.

Mean 8 days and SD 8 days (an exponential before censoring).
Used by every scenario except `F`.
The array is read-only.
"""
F_DEFAULT.flags.writeable = False
