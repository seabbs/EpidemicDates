# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""An example NumPyro model for a line list with missing onsets."""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike

from ._core import delay_pmf, linelist_loglik

__all__ = ["numpyro_linelist_model"]


def numpyro_linelist_model(
    complete: ArrayLike,
    missing_by_report: ArrayLike,
    delay_mean_prior: float = 5.0,
    delay_sd_prior: float = 5.0,
) -> None:
    r"""Fit onsets, delay and missingness jointly to a line list.

    An example of using `linelist_loglik` in NumPyro.
    It is a sketch to build on, not a tuned model.

    The onset curve is a random walk on the log scale,

    $$\log \lambda(0) \sim N(\log \bar{y}, 2), \qquad
    \log \lambda(s) = \log \lambda(s - 1) + \sigma \epsilon_s, \qquad
    \epsilon_s \sim N(0, 1), \qquad \sigma \sim N^+(0, 0.1),$$

    where $\bar{y}$ is the mean number of cases per day in the data.
    The delay is `delay_pmf(mu, sigma_d, D)` with
    $\mu \sim \mathrm{LogNormal}(\log \mu_0, 0.5)$ and
    $\sigma_d \sim \mathrm{LogNormal}(\log \sigma_0, 0.5)$.
    The probability that an onset is missing is constant,
    $\pi \sim \mathrm{Beta}(2, 2)$.
    The likelihood is `linelist_loglik`, added with `numpyro.factor`.

    Sample sites are `log_onset_init`, `rw_sd`, `rw_noise`,
    `delay_mean`, `delay_sd` and `missing_prob`.
    Deterministic sites are `onsets` ($\lambda$ on 0..T) and `delay`
    (the pmf on 0..D).

    Parameters
    ----------
    complete : array_like
        Complete-case counts by onset day and delay, shape
        (T + 1, D + 1), as from `tabulate_linelist`.
    missing_by_report : array_like
        Missing-onset counts by report day, length T + 1.
    delay_mean_prior : float, optional
        Prior median $\mu_0$ of the delay mean, in days.
    delay_sd_prior : float, optional
        Prior median $\sigma_0$ of the delay SD, in days.

    Examples
    --------
    >>> from jax import random
    >>> from numpyro.infer import MCMC, NUTS
    >>> arrays = tabulate_linelist(onset, report)  # doctest: +SKIP
    >>> mcmc = MCMC(NUTS(numpyro_linelist_model), num_warmup=300,
    ...             num_samples=300)  # doctest: +SKIP
    >>> mcmc.run(random.key(0), arrays.complete,
    ...          arrays.missing_by_report)  # doctest: +SKIP
    """
    try:
        import numpyro
        import numpyro.distributions as dist
    except ImportError as err:  # pragma: no cover - depends on the install
        raise ImportError(
            "numpyro_linelist_model needs NumPyro. Install the `jax` "
            'extra, for example `pip install "epidemic-dates[jax]"`.'
        ) from err

    complete = jnp.asarray(complete)
    missing_by_report = jnp.asarray(missing_by_report)
    n_days, n_delays = complete.shape
    per_day = (complete.sum() + missing_by_report.sum()) / n_days
    log_init = numpyro.sample(
        "log_onset_init",
        dist.Normal(jnp.log(jnp.maximum(per_day, 1.0)), 2.0),
    )
    rw_sd = numpyro.sample("rw_sd", dist.HalfNormal(0.1))
    rw_noise = numpyro.sample(
        "rw_noise", dist.Normal(0.0, 1.0).expand([n_days - 1]).to_event(1)
    )
    log_onsets = log_init + jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(rw_sd * rw_noise)]
    )
    onsets = numpyro.deterministic("onsets", jnp.exp(log_onsets))
    mean = numpyro.sample(
        "delay_mean", dist.LogNormal(jnp.log(delay_mean_prior), 0.5)
    )
    sd = numpyro.sample(
        "delay_sd", dist.LogNormal(jnp.log(delay_sd_prior), 0.5)
    )
    pmf = numpyro.deterministic("delay", delay_pmf(mean, sd, n_delays - 1))
    missing_prob = numpyro.sample("missing_prob", dist.Beta(2.0, 2.0))
    numpyro.factor(
        "linelist",
        linelist_loglik(onsets, pmf, missing_prob, complete, missing_by_report),
    )
