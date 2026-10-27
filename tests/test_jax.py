from collections.abc import Iterator

import numpy as np
import pytest
from scipy import stats

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")

import jax.numpy as jnp  # noqa: E402

from epidemic_dates import (  # noqa: E402
    DMAX,
    F_DEFAULT,
    backsampling_allocate,
    gamma_pmf,
    impute_linelist,
    scenario_setup,
    simulate,
)
from epidemic_dates import jax as ejax  # noqa: E402
from epidemic_dates.jax import (  # noqa: E402
    convolve_reports,
    delay_pmf,
    linelist_loglik,
    numpyro_linelist_model,
    reported_fraction,
    tabulate_linelist,
)

PARAMS = [(8, 8), (5, 2), (3, 6), (12, 12), (0.5, 0.3), (20, 3)]


@pytest.fixture
def x64() -> Iterator[None]:
    # Importing epidemic_dates.jax turns on float64; keep it on even if a
    # test switched it off.
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def test_import_turns_on_float64() -> None:
    assert jax.config.jax_enable_x64
    assert delay_pmf(8.0, 8.0).dtype == jnp.float64


@pytest.fixture(scope="module")
def linelist() -> dict[str, object]:
    setup = scenario_setup("C")
    onset, report, missing = simulate("C", setup, np.random.default_rng(1))
    cutoff = 80
    onset_obs = np.where(missing, np.nan, onset.astype(np.float64))
    arrays = tabulate_linelist(onset_obs, report, cutoff=cutoff)
    return {
        "setup": setup,
        "onset": onset_obs,
        "report": report,
        "cutoff": cutoff,
        "arrays": arrays,
        "truth": setup["O"][: cutoff + 1],
    }


@pytest.mark.parametrize(("mean", "sd"), PARAMS)
def test_delay_pmf_matches_numpy_in_float64(
    x64: None, mean: float, sd: float
) -> None:
    pmf = delay_pmf(mean, sd)
    assert pmf.dtype == jnp.float64
    assert pmf.shape == (DMAX + 1,)
    np.testing.assert_allclose(pmf, gamma_pmf(mean, sd), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(ejax.gamma_pmf(mean, sd), pmf)


@pytest.mark.parametrize(("mean", "sd"), [(8, 8), (5, 2), (12, 12)])
def test_delay_pmf_matches_numpy_in_float32(mean: float, sd: float) -> None:
    pmf = delay_pmf(jnp.float32(mean), jnp.float32(sd))
    assert pmf.dtype == jnp.float32
    # Cancellation in the second difference limits float32 to ~1e-5.
    np.testing.assert_allclose(pmf, gamma_pmf(mean, sd), atol=2e-5, rtol=0)
    assert float(pmf.sum()) == pytest.approx(1.0, abs=1e-6)


def test_delay_pmf_honours_dmax(x64: None) -> None:
    np.testing.assert_allclose(
        delay_pmf(8.0, 8.0, 20), gamma_pmf(8, 8, 20), atol=1e-12
    )


@pytest.mark.parametrize("use_x64", [False, True])
def test_delay_pmf_is_differentiable(use_x64: bool) -> None:
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", use_x64)
    try:
        weights = jnp.arange(DMAX + 1.0)

        def pmf_mean(mean: float, sd: float) -> jax.Array:
            return jnp.sum(weights * delay_pmf(mean, sd))

        def pmf_term(mean: float, sd: float) -> jax.Array:
            return delay_pmf(mean, sd)[0]

        for fun in (pmf_mean, pmf_term):
            grads = jax.jit(jax.grad(fun, argnums=(0, 1)))(8.0, 4.0)
            for grad in grads:
                assert np.isfinite(float(grad))
                assert float(grad) != 0.0
        # The discretised mean tracks the continuous one.
        d_mean = jax.grad(pmf_mean)(8.0, 4.0)
        assert float(d_mean) == pytest.approx(1.0, abs=0.05)
        # Gradients agree with finite differences of the NumPy pmf.
        eps = 1e-4
        fd = (gamma_pmf(8 + eps, 4)[10] - gamma_pmf(8 - eps, 4)[10]) / (2 * eps)
        ad = jax.grad(lambda m: delay_pmf(m, 4.0)[10])(8.0)
        assert float(ad) == pytest.approx(fd, rel=1e-3, abs=1e-6)
    finally:
        jax.config.update("jax_enable_x64", old)


def test_delay_pmf_jits(x64: None) -> None:
    fun = jax.jit(delay_pmf, static_argnums=2)
    np.testing.assert_allclose(fun(8.0, 8.0, DMAX), gamma_pmf(8, 8), atol=1e-12)


def test_convolve_reports_matches_numpy(x64: None) -> None:
    rng = np.random.default_rng(3)
    onsets = rng.gamma(2.0, 10.0, size=100)
    expected = np.convolve(onsets, F_DEFAULT)[: onsets.size]
    loop = np.array(
        [
            sum(onsets[t - d] * F_DEFAULT[d] for d in range(min(t, DMAX) + 1))
            for t in range(onsets.size)
        ]
    )
    got = np.asarray(jax.jit(convolve_reports)(onsets, F_DEFAULT))
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    np.testing.assert_allclose(got, loop, rtol=1e-12)


def test_convolve_reports_is_the_adjoint_of_backsampling(x64: None) -> None:
    # sum_t r(t) m(t) = sum_s lambda(s) (back-sampled m)(s).
    rng = np.random.default_rng(4)
    onsets = rng.gamma(2.0, 10.0, size=90)
    missing = rng.poisson(5.0, size=90).astype(np.float64)
    reports = np.asarray(convolve_reports(onsets, F_DEFAULT))
    back = backsampling_allocate(missing, F_DEFAULT)
    assert float(reports @ missing) == pytest.approx(float(onsets @ back))


@pytest.mark.parametrize("n_days", [5, 61, 100])
def test_backsampling_allocate_matches_numpy(x64: None, n_days: int) -> None:
    rng = np.random.default_rng(5)
    missing = rng.poisson(5.0, size=n_days).astype(np.float64)
    pmf = gamma_pmf(6, 4)
    got = jax.jit(ejax.backsampling_allocate)(missing, pmf)
    np.testing.assert_allclose(
        got, backsampling_allocate(missing, pmf), rtol=1e-12, atol=1e-12
    )


def test_reported_fraction_matches_numpy(x64: None) -> None:
    pmf = gamma_pmf(6, 4)
    for n_days, cutoff in [(100, 99), (100, 120), (30, 20), (5, 4)]:
        left = cutoff - np.arange(n_days)
        expected = np.where(
            left >= 0, np.cumsum(pmf)[np.clip(left, 0, DMAX)], 0.0
        )
        got = reported_fraction(jnp.asarray(pmf), n_days, cutoff)
        np.testing.assert_allclose(got, expected, rtol=1e-12)
    np.testing.assert_allclose(
        reported_fraction(pmf, 40), reported_fraction(pmf, 40, 39)
    )
    # Last onset day: only delay 0 is seen; far from the cut-off, all.
    frac = reported_fraction(pmf, 100, 99)
    assert float(frac[-1]) == pytest.approx(pmf[0])
    assert float(frac[0]) == pytest.approx(1.0)


def test_tabulate_linelist_matches_impute_linelist(
    linelist: dict[str, object],
) -> None:
    arrays = linelist["arrays"]
    cutoff = linelist["cutoff"]
    assert arrays.complete.shape == (cutoff + 1, DMAX + 1)
    assert arrays.missing_by_report.shape == (cutoff + 1,)
    assert arrays.cutoff == cutoff
    # No complete case is counted after the cut-off.
    s, d = np.nonzero(arrays.complete)
    assert np.all(s + d <= cutoff)
    res = impute_linelist(
        linelist["onset"], linelist["report"], cutoff=cutoff, n_iter=1
    )
    np.testing.assert_array_equal(arrays.complete.sum(1), res["complete"])
    np.testing.assert_array_equal(
        arrays.missing_by_report, res["missing_by_report"]
    )
    assert arrays.dropped == res["dropped"]


def test_tabulate_linelist_checks_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        tabulate_linelist([1, 2], [3])
    with pytest.raises(ValueError, match="report day"):
        tabulate_linelist([1, 2], [3, np.nan])
    with pytest.raises(ValueError, match="whole numbers"):
        tabulate_linelist([1.5, 2], [3, 4])
    with pytest.raises(ValueError, match="cutoff"):
        tabulate_linelist([1, 2], [3, 4], cutoff=-1)


def _brute_force_loglik(
    onsets: np.ndarray,
    pmf: np.ndarray,
    prob: np.ndarray,
    complete: np.ndarray,
    missing: np.ndarray,
) -> float:
    n_days = onsets.size
    total = 0.0
    for s in range(n_days):
        for d in range(pmf.size):
            t = s + d
            if t < n_days:
                mean = onsets[s] * pmf[d] * (1 - prob[t])
                total += stats.poisson.logpmf(complete[s, d], mean)
    reports = np.convolve(onsets, pmf)[:n_days]
    total += stats.poisson.logpmf(missing, prob * reports).sum()
    return float(total)


def test_linelist_loglik_matches_brute_force(
    x64: None, linelist: dict[str, object]
) -> None:
    arrays = linelist["arrays"]
    truth = linelist["truth"]
    n_days = truth.size
    prob = np.linspace(0.3, 0.5, n_days)
    got = jax.jit(linelist_loglik)(
        truth, F_DEFAULT, prob, arrays.complete, arrays.missing_by_report
    )
    expected = _brute_force_loglik(
        truth, F_DEFAULT, prob, arrays.complete, arrays.missing_by_report
    )
    assert float(got) == pytest.approx(expected, rel=1e-10)
    # A constant vector equals the scalar.
    args = (truth, F_DEFAULT)
    data = (arrays.complete, arrays.missing_by_report)
    assert float(
        linelist_loglik(*args, np.full(n_days, 0.4), *data)
    ) == pytest.approx(float(linelist_loglik(*args, 0.4, *data)))


def test_linelist_loglik_prefers_the_truth(
    x64: None, linelist: dict[str, object]
) -> None:
    arrays = linelist["arrays"]
    truth = jnp.asarray(linelist["truth"])
    data = (arrays.complete, arrays.missing_by_report)

    def loglik(
        onsets: jax.Array, mean: float, sd: float, prob: float
    ) -> jax.Array:
        return linelist_loglik(onsets, delay_pmf(mean, sd), prob, *data)

    best = float(loglik(truth, 8.0, 8.0, 0.4))
    assert np.isfinite(best)
    wrong = {
        "short delay": loglik(truth, 3.0, 3.0, 0.4),
        "long delay": loglik(truth, 16.0, 8.0, 0.4),
        "low missingness": loglik(truth, 8.0, 8.0, 0.1),
        "high missingness": loglik(truth, 8.0, 8.0, 0.7),
        "half the onsets": loglik(truth * 0.5, 8.0, 8.0, 0.4),
        "double the onsets": loglik(truth * 2.0, 8.0, 8.0, 0.4),
        "flat onsets": loglik(
            jnp.full_like(truth, truth.mean()), 8.0, 8.0, 0.4
        ),
    }
    for name, value in wrong.items():
        assert best > float(value), name


def test_linelist_loglik_gradients_are_finite(
    linelist: dict[str, object],
) -> None:
    arrays = linelist["arrays"]
    truth = jnp.asarray(linelist["truth"], dtype=jnp.float32)

    def loglik(log_onsets: jax.Array, mean: float, sd: float) -> jax.Array:
        return linelist_loglik(
            jnp.exp(log_onsets),
            delay_pmf(mean, sd),
            0.4,
            arrays.complete,
            arrays.missing_by_report,
        )

    grads = jax.jit(jax.grad(loglik, argnums=(0, 1, 2)))(
        jnp.log(truth), 8.0, 8.0
    )
    for grad in grads:
        assert np.all(np.isfinite(np.asarray(grad)))
    assert np.any(np.asarray(grads[0]) != 0)


@pytest.mark.slow
def test_numpyro_fit_recovers_the_truth(
    linelist: dict[str, object],
) -> None:
    from numpyro.infer import MCMC, NUTS

    arrays = linelist["arrays"]
    truth = linelist["truth"]
    cutoff = linelist["cutoff"]
    mcmc = MCMC(
        # A shallow tree keeps the run near 15 s with the same estimates.
        NUTS(numpyro_linelist_model, max_tree_depth=6),
        num_warmup=200,
        num_samples=200,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(0),
        arrays.complete,
        arrays.missing_by_report,
    )
    # Converting to NumPy waits for the asynchronous run to finish.
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    assert np.median(samples["delay_mean"]) == pytest.approx(8.0, abs=1.5)
    assert np.median(samples["delay_sd"]) == pytest.approx(8.0, abs=2.5)
    assert np.median(samples["missing_prob"]) == pytest.approx(0.4, abs=0.07)
    onsets = np.median(samples["onsets"], axis=0)
    assert onsets.shape == truth.shape
    assert np.corrcoef(onsets, truth)[0, 1] > 0.95
    # Away from the few cases at the start, the median tracks the truth.
    rel = np.abs(onsets / truth - 1)[10 : cutoff - 5]
    assert np.median(rel) < 0.15
    assert np.max(rel) < 0.5
