# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
r"""Delay pmf, convolution, truncation and line list likelihood in JAX.

All functions are pure and can be used under `jax.jit` and `jax.grad`.
Integer arguments (`dmax`, `n_days`, `cutoff`) set array shapes, so
they must be static (Python integers) under `jit`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import gammainc, gammaln
from jax.typing import ArrayLike

from ..delays import DMAX

__all__ = [
    "backsampling_allocate",
    "convolve_reports",
    "delay_pmf",
    "gamma_pmf",
    "linelist_loglik",
    "reported_fraction",
]


def delay_pmf(mean: ArrayLike, sd: ArrayLike, dmax: int = DMAX) -> Array:
    r"""Compute a double interval censored daily Gamma delay pmf in JAX.

    The JAX version of `epidemic_dates.gamma_pmf`.
    With onset uniform within its day and report censored to its day,

    $$P(d) = G(d + 1) - 2 G(d) + G(d - 1),$$

    $$G(x) = x F(x; k, \theta) - k \theta F(x; k + 1, \theta),$$

    where $F(x; k, \theta)$ is the Gamma CDF with shape
    $k = (\mu / \sigma)^2$ and scale $\theta = \sigma^2 / \mu$, and
    $G(x) = 0$ for $x \le 0$.
    $F$ is the regularised lower incomplete gamma function
    `jax.scipy.special.gammainc(k, x / theta)`, which is differentiable
    in $k$, so the pmf is differentiable in `mean` and `sd`.
    The pmf is floored at 0, truncated at `dmax` and renormalised.

    Parameters
    ----------
    mean : float or jax.Array
        Mean of the continuous Gamma delay, in days. Must be positive.
        Not checked, so it can be traced.
    sd : float or jax.Array
        Standard deviation of the continuous Gamma delay, in days. Must
        be positive. Not checked.
    dmax : int, optional
        Maximum delay. Static under `jit`. Defaults to `DMAX`.

    Returns
    -------
    jax.Array
        Probabilities for delays 0..dmax, of length `dmax + 1`, summing
        to 1.
    """
    mean = jnp.asarray(mean)
    sd = jnp.asarray(sd)
    dtype = jnp.result_type(mean, sd, float)
    shape = (mean / sd) ** 2
    scale = sd**2 / mean

    def integrated_cdf(x: Array) -> Array:
        # Keep the argument positive so gradients at x <= 0 are finite.
        pos = x > 0
        x_safe = jnp.where(pos, x, 1.0)
        value = x_safe * gammainc(shape, x_safe / scale) - (
            shape * scale * gammainc(shape + 1, x_safe / scale)
        )
        return jnp.where(pos, value, 0.0)

    days = jnp.arange(dmax + 1, dtype=dtype)
    pmf = jnp.maximum(
        integrated_cdf(days + 1)
        - 2 * integrated_cdf(days)
        + integrated_cdf(days - 1),
        0.0,
    )
    return pmf / pmf.sum()


gamma_pmf = delay_pmf
"""Alias of `delay_pmf`, named as `epidemic_dates.gamma_pmf`."""


def convolve_reports(onsets: ArrayLike, pmf: ArrayLike) -> Array:
    r"""Compute expected reports by report day from an onset curve.

    $$r(t) = \sum_{d=0}^{\min(t, D)} \lambda(t - d) f(d),
    \qquad t = 0, \dots, S - 1,$$

    where $S$ is the length of `onsets` and $D + 1$ the length of
    `pmf`.
    Onsets before day 0 are taken as 0.

    Parameters
    ----------
    onsets : jax.Array
        Expected onsets $\lambda(s)$ on onset days 0..S-1.
    pmf : jax.Array
        Delay pmf $f(d)$ on 0..D.

    Returns
    -------
    jax.Array
        Expected reports $r(t)$ on report days 0..S-1, the same length
        as `onsets`.
    """
    onsets = jnp.asarray(onsets)
    pmf = jnp.asarray(pmf)
    return jnp.convolve(onsets, pmf, mode="full")[: onsets.shape[0]]


def backsampling_allocate(
    missing_by_report: ArrayLike, delay_pmf: ArrayLike
) -> Array:
    r"""Allocate missing-onset cases back from report day by a delay pmf.

    The JAX version of `epidemic_dates.backsampling_allocate`:

    $$a(s) = \sum_{d=0}^{D} m(s + d) f(d), \qquad s = 0, \dots, T,$$

    with $m(t) = 0$ for $t > T$.
    It is the adjoint (transpose) of `convolve_reports`.

    Parameters
    ----------
    missing_by_report : jax.Array
        Counts of missing-onset cases $m(t)$ by report day 0..T.
    delay_pmf : jax.Array
        Delay pmf $f(d)$ on 0..D.

    Returns
    -------
    jax.Array
        Expected missing-onset cases by onset day 0..T. Mass that falls
        before day 0 is not included.
    """
    missing = jnp.asarray(missing_by_report)
    pmf = jnp.asarray(delay_pmf)
    n_days = missing.shape[0]
    # Correlate: reverse, convolve, reverse.
    full = jnp.convolve(missing[::-1], pmf, mode="full")[:n_days]
    return full[::-1]


def reported_fraction(
    pmf: ArrayLike, n_days: int, cutoff: int | None = None
) -> Array:
    r"""Compute the fraction of each onset day reported by the cut-off.

    $$F(T - s) = \sum_{d=0}^{\min(T - s, D)} f(d),
    \qquad s = 0, \dots, n - 1,$$

    with $F(x) = 0$ for $x < 0$, where $T$ is `cutoff` and $n$ is
    `n_days`.
    This is the right-truncation factor used in the M-step of
    `epidemic_dates.em_fit`.

    Parameters
    ----------
    pmf : jax.Array
        Delay pmf $f(d)$ on 0..D.
    n_days : int
        Number of onset days. Static under `jit`.
    cutoff : int, optional
        Data cut-off day T. Defaults to `n_days - 1`.

    Returns
    -------
    jax.Array
        $F(T - s)$ for onset days 0..n_days-1.
    """
    pmf = jnp.asarray(pmf)
    if cutoff is None:
        cutoff = n_days - 1
    cdf = jnp.cumsum(pmf)
    left = cutoff - jnp.arange(n_days)
    idx = jnp.clip(left, 0, pmf.shape[0] - 1)
    return jnp.where(left >= 0, cdf[idx], 0.0)


def linelist_loglik(
    onsets: ArrayLike,
    pmf: ArrayLike,
    missing_prob: ArrayLike,
    complete: ArrayLike,
    missing_by_report: ArrayLike,
) -> Array:
    r"""Log-likelihood of a right-truncated line list with missing onsets.

    Onset days are $s = 0, \dots, T$ and report days
    $t = 0, \dots, T$, where $T$ is the cut-off and $T + 1$ the length
    of `onsets`.
    Delays are $d = 0, \dots, D$ with pmf $f(d)$.
    Each case's onset is missing with probability $\pi(t)$, which may
    depend on its report day $t$ (missing at random given report day).

    Complete cases, counted by onset day and delay as $n(s, d)$, are
    Poisson with mean

    $$\mu_c(s, d) = \lambda(s) f(d) \{1 - \pi(s + d)\},$$

    observed only when $s + d \le T$ (right truncation).
    Missing-onset cases, counted by report day as $m(t)$, are Poisson
    with mean

    $$\mu_m(t) = \pi(t) \sum_{s=\max(0, t - D)}^{t} \lambda(s) f(t - s).$$

    The two sets of counts are thinned from the same Poisson process,
    so they are independent and

    $$\ell = \sum_{s + d \le T} \{n(s, d) \log \mu_c(s, d)
    - \mu_c(s, d) - \log n(s, d)!\}
    + \sum_{t=0}^{T} \{m(t) \log \mu_m(t) - \mu_m(t) - \log m(t)!\}.$$

    Cases with onset before day 0 are assumed absent.
    Means are floored at the smallest positive float before taking
    logs, so zero counts in cells with zero mean give finite gradients.

    Parameters
    ----------
    onsets : jax.Array
        Expected onsets $\lambda(s)$ of all cases, reported or not, on
        onset days 0..T. Must be positive.
    pmf : jax.Array
        Delay pmf $f(d)$ on 0..D.
    missing_prob : float or jax.Array
        Probability $\pi$ that an onset is missing. A scalar, or an
        array of length T + 1 indexed by report day.
    complete : jax.Array
        Complete-case counts $n(s, d)$, shape (T + 1, D + 1). Cells with
        $s + d > T$ are ignored.
    missing_by_report : jax.Array
        Missing-onset counts $m(t)$ by report day, length T + 1.

    Returns
    -------
    jax.Array
        Scalar log-likelihood.

    See Also
    --------
    tabulate_linelist : Build `complete` and `missing_by_report` from a
        line list.
    """
    onsets = jnp.asarray(onsets)
    pmf = jnp.asarray(pmf)
    complete = jnp.asarray(complete)
    missing_by_report = jnp.asarray(missing_by_report)
    n_days = onsets.shape[0]
    n_delays = pmf.shape[0]
    dtype = jnp.result_type(onsets, pmf, float)
    tiny = np.finfo(dtype).tiny
    prob = jnp.broadcast_to(jnp.asarray(missing_prob, dtype=dtype), (n_days,))

    report = jnp.arange(n_days)[:, None] + jnp.arange(n_delays)[None, :]
    observed = report < n_days
    prob_cell = prob[jnp.minimum(report, n_days - 1)]
    mean_c = onsets[:, None] * pmf[None, :] * (1 - prob_cell)
    mean_c = jnp.maximum(mean_c, tiny)
    ll_c = complete * jnp.log(mean_c) - mean_c - gammaln(complete + 1)
    ll_complete = jnp.sum(jnp.where(observed, ll_c, 0.0))

    mean_m = jnp.maximum(prob * convolve_reports(onsets, pmf), tiny)
    ll_missing = jnp.sum(
        missing_by_report * jnp.log(mean_m)
        - mean_m
        - gammaln(missing_by_report + 1)
    )
    return ll_complete + ll_missing
