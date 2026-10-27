# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Simulation study: bias of onset-date imputation for missing onsets.

Methods for cases with a missing onset (complete cases keep their true
onset):

- `hub`: onset = report - delay drawn from the pooled empirical delays of
  complete cases reported by cut-off T.
- `mean`: onset = report - round(mean empirical delay).
- `forward`: onset = report - delay drawn from the true forward delay pmf.
- `em`: EM back-projection (`em_backproject`). p(s | t) is proportional
  to O(s) f(t - s), with onsets O and a static delay f estimated jointly
  from all cases with right truncation at T. Imputed counts are the
  expected allocation.

Command line: `uv run onset-imputation --reps 200 --out output`.
From Python: see `run_scenario` and `run_study`.

Days are integers. Day 0 is the first possible onset in a scenario.
"""

import argparse
import csv
import os

import numpy as np
from scipy import stats

DMAX = 60
"""Maximum onset-to-report delay in days. Delay pmfs cover 0..DMAX."""
METHODS = ["hub", "mean", "forward", "em"]
"""Imputation methods compared, in the order used in summary rows."""
OFF = 70
"""Day offset so negative imputed onsets bin correctly into weeks."""


def gamma_pmf(mean, sd, dmax=DMAX):
    r"""Double interval censored daily Gamma delay pmf.

    Uses the primarycensored form. The primary event (onset) is uniform
    within its day and the secondary event (report) is censored to its
    day, so

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
        Mean of the continuous Gamma delay, in days.
    sd : float
        Standard deviation of the continuous Gamma delay, in days.
    dmax : int, optional
        Maximum delay. Defaults to `DMAX`.

    Returns
    -------
    numpy.ndarray
        Probabilities for delays 0..dmax, of length `dmax + 1`, summing
        to 1.
    """
    shape = (mean / sd) ** 2
    scale = sd**2 / mean

    def g(x):
        x = np.maximum(x, 0.0)
        return x * stats.gamma.cdf(x, shape, scale=scale) - (
            shape * scale * stats.gamma.cdf(x, shape + 1, scale=scale)
        )

    d = np.arange(dmax + 1, dtype=float)
    p = np.maximum(g(d + 1) - 2 * g(d) + g(d - 1), 0.0)
    return p / p.sum()


F_DEFAULT = gamma_pmf(8, 8)
"""Default forward delay pmf: `gamma_pmf(8, 8)`, mean 8 days, SD 8 days
(an exponential before censoring), on 0..DMAX. Used by every scenario
except `F`.
"""


def r_path(kind, L):
    """Daily growth rate r(t) for scenario `kind` over days 0..L-1.

    Raises `ValueError` for an unknown scenario.
    """
    t = np.arange(L, dtype=float)
    if kind == "A":
        return np.full(L, 0.05)
    if kind == "B":
        return np.where(t < 60, 0.05, -0.03)
    if kind == "B0":
        return np.zeros(L)
    if kind in ("C", "G", "H", "I", "Csmall"):
        return np.interp(t, [0, 60, 100, L], [0.05, 0.05, -0.02, -0.02])
    if kind == "D":
        return np.where(t < 80, 0.06, -0.04)
    if kind == "E":
        return np.full(L, 0.008)
    if kind == "F":
        return np.full(L, 0.03)
    if kind == "W":
        return np.interp(
            t,
            [0, 30, 50, 90, 105, 135, 150],
            [0.06, 0.06, -0.03, -0.03, 0.05, 0.05, -0.04],
        )
    raise ValueError(kind)


SCEN = {
    "A": (120, 10000, None),
    "B": (150, 5000, 8),  # switch at day 60
    "B0": (150, 4000, None),
    "C": (150, 5000, 12),  # r crosses 0 at day ~89
    "D": (150, 5000, 11),  # switch at day 80
    "E": (150, 3000, None),
    "F": (150, 5000, None),
    "G": (150, 5000, 12),
    "H": (150, 5000, 12),
    "I": (150, 5000, 10),  # ascertainment doubles at day 70
    "Csmall": (150, 300, 12),
    "W": (150, 5000, 7),  # first turn at day ~40
}
"""Scenario settings, keyed by name. Each value is
`(L, target, cp)`: the number of onset days, the expected total number
of cases and the change-point week (or `None`). The change-point week
is used for the `*_cp` summary columns. The growth rate path is set in
`r_path` and missingness in `miss_prob`. See `DESC`.
"""

DESC = {
    "A": "r=0.05 constant",
    "B": "r=0.05 then -0.03 at d60",
    "B0": "r=0 stable",
    "C": "r 0.05->-0.02 over d60-100",
    "D": "r 0.06->-0.04 at d80",
    "E": "r=0.008 (Isiro-like)",
    "F": "r=0.03, mean delay 12->4",
    "G": "as C, P(miss) 0.2->0.7 with delay",
    "H": "as C, P(miss) 0.2->0.6 by report d60-100",
    "I": "as C, ascertainment x2 at d70",
    "Csmall": "as C, ~300 cases",
    "W": "two waves: growth, control, resurgence, control",
}
"""One-line description of each scenario in `SCEN`. Unless stated, the
delay is `F_DEFAULT` and each onset is missing with probability 0.4.
"""


def scenario_setup(name):
    """Build the expected onset curve and delays for a scenario.

    Parameters
    ----------
    name : str
        A key of `SCEN`.

    Returns
    -------
    dict
        Scenario inputs with keys:

        - `L`: number of onset days.
        - `O`: expected daily onsets, shape (L,), summing to the target
          total. Built as exp(cumulative sum of r(t)) and rescaled. In
          scenario `I` it is doubled from onset day 70 before rescaling.
        - `fmat`: forward delay pmf by onset day, shape (L, DMAX + 1).
          In scenario `F` the mean (and SD) falls linearly from 12 to 4
          days; otherwise every row is `F_DEFAULT`.
        - `cp`: change-point week or `None`.
    """
    L, target, cp = SCEN[name]
    r = r_path(name, L)
    shape = np.exp(np.concatenate([[0], np.cumsum(r[1:])]))
    if name == "I":
        shape = shape * np.where(np.arange(L) >= 70, 2.0, 1.0)
    O = shape * target / shape.sum()
    if name == "F":
        means = np.linspace(12, 4, L)
        fmat = np.array([gamma_pmf(m, m) for m in means])
    else:
        fmat = np.tile(F_DEFAULT, (L, 1))
    return {"L": L, "O": O, "fmat": fmat, "cp": cp}


def miss_prob(name, delay, report):
    """Probability that each case's onset is missing.

    0.4 for all cases, except in `G` (0.2 rising to 0.7 as the delay
    goes from 0 to 20 days) and `H` (0.2 rising to 0.6 as the report day
    goes from 60 to 100).
    """
    if name == "G":
        return 0.2 + 0.5 * np.clip(delay / 20, 0, 1)
    if name == "H":
        return np.interp(report, [0, 60, 100, 1e9], [0.2, 0.2, 0.6, 0.6])
    return np.full(delay.shape, 0.4)


def simulate(name, sp, rng):
    """Simulate one line list for a scenario.

    Daily onset counts are Poisson with mean `sp["O"]`. Each case gets a
    delay from its onset day's forward pmf, a report day of onset plus
    delay, and a missing-onset flag from `miss_prob`.

    Parameters
    ----------
    name : str
        A key of `SCEN`.
    sp : dict
        Output of `scenario_setup(name)`.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    onset, report, missing : numpy.ndarray
        Integer onset and report days and a boolean missing-onset flag,
        one entry per case. No cut-off is applied; reports run to
        `L - 1 + DMAX`.
    """
    n = rng.poisson(sp["O"])
    onset = np.repeat(np.arange(sp["L"]), n)
    if name == "F":
        cdfs = np.cumsum(sp["fmat"], axis=1)
        u = rng.random(onset.size)
        delay = (u[:, None] > cdfs[onset]).sum(axis=1)
        delay = np.minimum(delay, DMAX)
    else:
        delay = rng.choice(DMAX + 1, size=onset.size, p=F_DEFAULT)
    report = onset + delay
    missing = rng.random(onset.size) < miss_prob(name, delay, report)
    return onset, report, missing


def em_backproject(sc, rm, T, n_iter=150):
    r"""Back-project missing-onset cases by EM with an estimated delay.

    Expected counts by onset day $s$ and delay $d$ are
    $\lambda(s) f(d)$, observed only when $s + d \le T$ (right
    truncation). Complete cases give counts $n_c(s, d)$. Missing cases
    give counts $m(t)$ by report day only.

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

    where $F$ is the delay CDF and $D$ = `DMAX`. $f$ is floored at
    $10^{-12}$ and renormalised.

    The delay is static over time. The method assumes onsets are missing
    at random given the report day. No smoothing is applied to
    $\lambda$.

    Parameters
    ----------
    sc : tuple of numpy.ndarray
        `(s_c, d_c)`: integer onset days and delays of complete cases.
        Onset days must lie in 0..T and delays in 0..DMAX, with
        onset plus delay <= T.
    rm : numpy.ndarray
        Integer report days (0..T) of cases with a missing onset.
    T : int
        Data cut-off day.
    n_iter : int, optional
        Number of EM iterations. No convergence check is made.

    Returns
    -------
    numpy.ndarray
        Expected number of missing-onset cases on each onset day 0..T,
        length `T + 1`. Sums to `len(rm)`. The estimated onset curve and
        delay are not returned.
    """
    s_c, d_c = sc
    S = T + 1
    Nc = np.zeros((S, DMAX + 1))
    np.add.at(Nc, (s_c, d_c), 1.0)
    M = np.bincount(rm, minlength=S)[:S].astype(float)
    ss, dd = np.meshgrid(np.arange(S), np.arange(DMAX + 1), indexing="ij")
    tt = ss + dd
    obs = tt <= T
    tt_c = np.where(obs, tt, 0)
    f = Nc.sum(0) + 0.5
    f /= f.sum()
    O = Nc.sum(1) / max(1e-9, 1 - M.sum() / (M.sum() + Nc.sum())) + 0.1
    A = np.zeros_like(Nc)
    for _ in range(n_iter):
        P = O[:, None] * f[None, :] * obs
        den = np.bincount(tt_c.ravel(), weights=P.ravel(), minlength=S)
        A = np.where(obs, P * (M / np.maximum(den, 1e-300))[tt_c], 0.0)
        N = Nc + A
        # M-step: right-truncated triangle (alternating updates)
        for _ in range(3):
            Fcum = np.cumsum(f)
            O = N.sum(1) / Fcum[np.minimum(T - np.arange(S), DMAX)]
            cO = np.cumsum(O)
            dr = np.arange(DMAX + 1)
            exposure = np.where(T - dr >= 0, cO[np.maximum(T - dr, 0)], 0.0)
            f = N.sum(0) / np.maximum(exposure, 1e-300)
            f = np.maximum(f, 1e-12)
            f /= f.sum()
    return A.sum(1)


def impute_linelist(onset, report, cutoff=None, n_iter=150):
    """Impute missing onsets in a real line list.

    Parameters
    ----------
    onset : array_like
        Onset day per case, with NaN (or None) where missing.
    report : array_like of int
        Report day per case. Must have no missing values. Days are
        integers, for example days since the first report.
    cutoff : int, optional
        Data cut-off day T. Defaults to the last report day. Cases
        reported before day 0 or after T are dropped.
    n_iter : int, optional
        Number of EM iterations passed to `em_backproject`.

    Returns
    -------
    dict
        Daily arrays have length T + 1 and index i is onset day i.

        - `complete`: onset counts of cases with a known onset.
        - `missing_by_report`: counts of missing-onset cases by report
          day.
        - `empirical_delay`: pooled delay pmf (0..DMAX) of complete
          cases.
        - `imputed`: dict of daily onset counts (complete plus imputed)
          for `hub`, `mean` and `em`.
        - `dropped`: number of cases dropped.

    Notes
    -----
    - `hub` spreads each missing case over report day minus each delay,
      weighted by `empirical_delay`. It is the expected allocation of the
      random draw, not a draw.
    - `mean` shifts each missing case back by the rounded mean of
      `empirical_delay`.
    - `em` uses `em_backproject`. It estimates a static delay jointly
      with the onset curve, allowing for right truncation at the cut-off.
    - All methods assume onsets are missing at random given the report
      day. If missingness depends on the delay, all are biased.
    - Complete cases with a negative onset, an onset after report or a
      delay over `DMAX` are dropped. They are not imputed.
    - For `hub` and `mean`, imputed onsets before day 0 are discarded,
      so totals can be below the number of cases kept.
    - Outputs are expected counts and need not be whole numbers.
      Sum to weeks before comparing methods.
    """
    onset = np.asarray(onset, dtype=float)
    report = np.asarray(report, dtype=int)
    T = int(report.max()) if cutoff is None else int(cutoff)
    keep = (report >= 0) & (report <= T)
    onset, report = onset[keep], report[keep]
    missing = np.isnan(onset)
    s_c = onset[~missing].astype(int)
    d_c = report[~missing] - s_c
    ok = (s_c >= 0) & (d_c >= 0) & (d_c <= DMAX)
    s_c, d_c = s_c[ok], d_c[ok]
    rm = report[missing]
    S = T + 1
    comp = np.bincount(s_c, minlength=S)[:S].astype(float)
    miss = np.bincount(rm, minlength=S)[:S].astype(float)
    g = np.bincount(d_c, minlength=DMAX + 1).astype(float)
    g /= g.sum()
    hub = comp.copy()
    for d in range(DMAX + 1):
        hub[: S - d] += miss[d:] * g[d]
    m = round(float(np.dot(np.arange(DMAX + 1), g)))
    mean = comp.copy()
    mean[: S - m] += miss[m:]
    em = comp + em_backproject((s_c, d_c), rm, T, n_iter=n_iter)
    return {
        "complete": comp,
        "missing_by_report": miss,
        "empirical_delay": g,
        "imputed": {"hub": hub, "mean": mean, "em": em},
        "dropped": int((~keep).sum() + (~ok).sum()),
    }


def weekly(days, weights=None, nweeks=40):
    """Sum (weighted) days into weeks 0..nweeks-1; week 0 is days 0-6.

    Days before 0 (down to -OFF) or after the last week are discarded.
    """
    idx = (np.asarray(days) + OFF) // 7
    w = np.bincount(idx, weights=weights, minlength=nweeks + OFF // 7)
    return w[OFF // 7 : OFF // 7 + nweeks]


def run_rep(name, sp, rng, cutoffs, fwd_draw_f):
    """Simulate one line list and impute it at each cut-off.

    Returns a dict keyed by cut-off. Each value holds weekly onset
    counts for `truth` (cases reported by the cut-off) and each of
    `METHODS`, plus `n`, the number of cases reported by the cut-off.
    """
    onset, report, missing = simulate(name, sp, rng)
    out = {}
    for T in cutoffs:
        k = report <= T
        on, rep, mi = onset[k], report[k], missing[k]
        res = {"truth": weekly(on)}
        comp_on = on[~mi]
        comp_d = (rep - on)[~mi]
        rm = rep[mi]
        base = weekly(comp_on)
        emp = comp_d
        res["hub"] = base + weekly(rm - rng.choice(emp, size=rm.size))
        res["mean"] = base + weekly(rm - round(emp.mean()))
        if fwd_draw_f is None:
            fd = rng.choice(DMAX + 1, size=rm.size, p=F_DEFAULT)
        else:
            # time-varying forward pmf, indexed by report day
            fm = fwd_draw_f[np.minimum(rm, sp["L"] - 1)]
            u = rng.random(rm.size)
            fd = np.minimum((u[:, None] > np.cumsum(fm, 1)).sum(1), DMAX)
        res["forward"] = base + weekly(rm - fd)
        alloc = em_backproject((comp_on, comp_d), rm, T)
        res["em"] = base + weekly(np.arange(T + 1), weights=alloc)
        res["n"] = k.sum()
        out[T] = res
    return out


def gr(y):
    """Centred 3-week log-linear slope (per day); NaN at ends."""
    ly = np.log(y + 0.5)
    g = np.full(y.shape, np.nan)
    g[..., 1:-1] = (ly[..., 2:] - ly[..., :-2]) / 2 / 7
    return g


def first_neg(g, start=4):
    """First week from `start` with two consecutive negative growth rates.

    One value per row of `g`; NaN if there is none.
    """
    out = np.full(g.shape[0], np.nan)
    for i in range(g.shape[0]):
        for w in range(start, g.shape[1] - 2):
            if g[i, w] < 0 and g[i, w + 1] < 0:
                out[i] = w
                break
    return out


def summarise(name, sp, T, reps, retro):
    """Summarise each method's bias over replicates at one cut-off.

    Parameters
    ----------
    name : str
        A key of `SCEN`.
    sp : dict
        Output of `scenario_setup(name)`.
    T : int
        Cut-off day.
    reps : list of dict
        Outputs of `run_rep`.
    retro : bool
        Whether T is the retrospective cut-off. If so, weeks 0..L//7-1
        are scored. Otherwise the complete weeks up to T are scored.

    Returns
    -------
    list of dict
        One row per method. See `run_scenario` for the columns.
    """
    L, cp = sp["L"], sp["cp"]
    nw = (L // 7) if retro else (T + 1) // 7
    truth = np.array([r[T]["truth"][:nw] for r in reps])
    ntot = np.mean([r[T]["n"] for r in reps])
    mt = truth.mean(0)
    thr = 3 if name == "Csmall" else 10
    inc = mt >= thr
    wk = np.arange(nw)
    cpw = inc & (np.abs(wk - cp) <= 3) if cp is not None else None
    gt = gr(truth)
    rows = []
    for m in METHODS:
        est = np.array([r[T][m][:nw] for r in reps])
        rb = est.mean(0) / np.where(mt > 0, mt, np.nan) - 1
        rel_rep = (est - truth) / np.maximum(truth, 1)
        rmse = np.sqrt(np.nanmean(rel_rep[:, inc] ** 2))
        ge = gr(est)
        gb = np.nanmean(ge - gt, axis=0)
        ginc = inc.copy()
        ginc[[0, -1]] = False
        d = {
            "scenario": name,
            "desc": DESC[name],
            "cutoff": T,
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
            d["relbias_cp"] = np.nanmean(rb[cpw])
            g_cp = cpw & ginc
            d["gr_bias_cp"] = np.nanmean(gb[g_cp])
            d["gr_bias_cp_maxabs"] = np.nanmax(np.abs(gb[g_cp]))
        if retro:
            pk_t = truth.argmax(1)
            pk_e = est.argmax(1)
            d["peak_shift_wk"] = np.mean(pk_e - pk_t)
            d["peak_diff_frac"] = np.mean(pk_e != pk_t)
            if cp is not None:
                zt, ze = first_neg(gt), first_neg(ge)
                ok = ~np.isnan(zt) & ~np.isnan(ze)
                d["zero_cross_lag_wk"] = np.mean(ze[ok] - zt[ok])
            # mean curve peak
            d["peak_meancurve_shift"] = int(est.mean(0).argmax() - mt.argmax())
        else:
            d["relbias_last2"] = np.nanmean(rb[nw - 2 : nw])
            d["relbias_last4"] = np.nanmean(rb[nw - 4 : nw])
            d["gr_bias_last"] = gb[nw - 2]
        rows.append(d)
    return rows


def analytic():
    """Analytic checks of pooled complete-case delays.

    Compares the forward delay pmf with the backward pmf under constant
    growth, and checks the result by simulation.

    Returns
    -------
    list of str
        Lines of text:

        - for r = 0.05, 0.008 and -0.03, the forward and backward mean
          delays, the retrospective level factor of `hub` imputed cases,
          E[exp(-rD)] E[exp(rD)], and of `mean` imputed cases,
          exp(rm) E[exp(-rD)], with D ~ `F_DEFAULT` and m its rounded
          mean;
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
        Mr = (f * np.exp(-r * d)).sum()
        Mmr = (f * np.exp(r * d)).sum()
        b = f * np.exp(-r * d) / Mr
        lines.append(
            f"r={r:+.3f}: forward mean {np.dot(d, f):.2f}, backward mean "
            f"{np.dot(d, b):.2f}; retrospective hub level factor for "
            f"imputed cases E[e^-rD]E[e^rD]={Mr * Mmr:.3f}; mean-shift "
            f"factor e^(r m)E[e^-rD]={np.exp(r * round(np.dot(d, f))) * Mr:.3f}"
        )
    # empirical check under scenario A
    rng = np.random.default_rng(1)
    sp = scenario_setup("A")
    on, rep, mi = simulate("A", sp, rng)
    for T in (90, 179):
        k = (rep <= T) & ~mi
        dd = rep[k] - on[k]
        lines.append(
            f"Scenario A sim, T={T}: pooled complete-case mean delay "
            f"{dd.mean():.2f}"
        )
    # pooled truncated pmf for growth since day 0, T = 90
    T, r = 90, 0.05
    w = np.array([(np.exp(r * np.arange(0, T - x + 1))).sum() for x in d])
    p = f * w / (f * w).sum()
    lines.append(
        f"Analytic pooled truncated pmf (A, T=90) mean "
        f"{np.dot(d, p):.2f}; TV distance to backward "
        f"{0.5 * np.abs(p - f * np.exp(-r * d) / (f * np.exp(-r * d)).sum()).sum():.4f}"
    )
    return lines


def run_scenario(name, reps=200, seed=20261027, cutoffs=None):
    """Simulate one scenario and summarise each method's bias.

    Parameters
    ----------
    name : str
        A key of `SCEN` (see `DESC`).
    reps : int, optional
        Number of stochastic replicates.
    seed : int, optional
        Seed for `numpy.random.default_rng`.
    cutoffs : list of int, optional
        Data cut-off days. The retrospective cut-off is `L - 1 + DMAX`,
        when every case has been reported. Defaults to the retrospective
        cut-off plus real-time cut-offs 90 and 120 (70 and 90 for `A`).
        A cut-off is scored as retrospective only if it equals that day.

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
          bias in the weekly growth rate (per day). The growth rate is the
          centred log-linear slope over three weeks of log(count + 0.5).

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
    """
    rng = np.random.default_rng(seed)
    sp = scenario_setup(name)
    retro_t = sp["L"] - 1 + 60
    if cutoffs is None:
        cutoffs = [retro_t, 90, 120] if name != "A" else [retro_t, 70, 90]
    fwd = sp["fmat"] if name == "F" else None
    sims = [run_rep(name, sp, rng, cutoffs, fwd) for _ in range(reps)]
    rows = []
    for T in cutoffs:
        rows += summarise(name, sp, T, sims, T == retro_t)
    return rows


def run_study(reps=200, seed=20261027, scenarios=None, verbose=True):
    """Run every scenario, or those named, and return all summary rows.

    Parameters
    ----------
    reps : int, optional
        Number of replicates per scenario.
    seed : int, optional
        Base seed. The i-th scenario run uses `seed + i`, so results for
        a scenario depend on its position in `scenarios`.
    scenarios : list of str, optional
        Keys of `SCEN`. Defaults to all scenarios.
    verbose : bool, optional
        Print a line as each scenario finishes.

    Returns
    -------
    list of dict
        Rows from `run_scenario` for each scenario.
    """
    rows = []
    for i, name in enumerate(scenarios or SCEN):
        rows += run_scenario(name, reps=reps, seed=seed + i)
        if verbose:
            print(f"done {name}", flush=True)
    return rows


def write_results(rows, out):
    """Write summary rows and analytic checks to a directory.

    Parameters
    ----------
    rows : list of dict
        Summary rows, for example from `run_study`.
    out : str
        Output directory. Created if needed.

    Notes
    -----
    Writes `out/results.csv`, with columns in order of first appearance
    and floats to 4 decimal places, and `out/analytic.txt`, the output
    of `analytic`. Existing files are overwritten.
    """
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: (f"{v:.4f}" if isinstance(v, float) else v)
                    for k, v in r.items()
                }
            )
    with open(os.path.join(out, "analytic.txt"), "w") as fh:
        fh.write("\n".join(analytic()) + "\n")


def main(argv=None):
    """Command-line entry point (`onset-imputation`).

    Runs `run_study`, writes results with `write_results` and prints the
    analytic checks.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. Defaults to `sys.argv[1:]`. Options are `--reps`
        (default 200), `--seed` (default 20261027), `--out` (default
        `output`) and `--scenarios` (default all).
    """
    parser = argparse.ArgumentParser(
        description="Simulation study of onset-date imputation bias."
    )
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--out", default="output")
    parser.add_argument(
        "--scenarios", nargs="*", choices=list(SCEN), default=None
    )
    args = parser.parse_args(argv)
    rows = run_study(args.reps, args.seed, args.scenarios)
    write_results(rows, args.out)
    print("\n".join(analytic()))


if __name__ == "__main__":
    main()
