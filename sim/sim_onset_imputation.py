"""Simulation study: bias of onset-date imputation for missing onsets.

Methods for cases with missing onset (complete cases keep true onset):
  hub     : onset = report - delay drawn from pooled empirical delays of
            complete cases reported by cut-off T
  mean    : onset = report - round(mean empirical delay)
  forward : onset = report - delay drawn from the true forward delay pmf
  em      : EM back-projection; p(s | t) prop. to O(s) f(t - s), with O and
            a static f estimated jointly from all cases with right
            truncation at T; imputed counts are the expected allocation

Run with: uv run sim/sim_onset_imputation.py [replicates]
Results are written to sim/output/.
"""

import csv
import os
import sys

import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DMAX = 60
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 200
METHODS = ["hub", "mean", "forward", "em"]
OFF = 70  # day offset so negative imputed onsets bin correctly


def gamma_pmf(mean, sd, dmax=DMAX):
    shape = (mean / sd) ** 2
    scale = sd**2 / mean
    cdf = stats.gamma.cdf(np.arange(dmax + 2), shape, scale=scale)
    p = np.diff(cdf)
    return p / p.sum()


F_DEFAULT = gamma_pmf(8, 8)


def r_path(kind, L):
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


# name: (L, target expected total, change-point week or None)
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


def scenario_setup(name):
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
    if name == "G":
        return 0.2 + 0.5 * np.clip(delay / 20, 0, 1)
    if name == "H":
        return np.interp(report, [0, 60, 100, 1e9], [0.2, 0.2, 0.6, 0.6])
    return np.full(delay.shape, 0.4)


def simulate(name, sp, rng):
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
    """EM back-projection with joint static delay estimation.

    sc, dc: complete onset days and delays; rm: missing report days.
    Returns expected daily onset allocation of missing cases (0..T).
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


def weekly(days, weights=None, nweeks=40):
    idx = (np.asarray(days) + OFF) // 7
    w = np.bincount(idx, weights=weights, minlength=nweeks + OFF // 7)
    return w[OFF // 7 : OFF // 7 + nweeks]


def run_rep(name, sp, rng, cutoffs, fwd_draw_f):
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
    out = np.full(g.shape[0], np.nan)
    for i in range(g.shape[0]):
        for w in range(start, g.shape[1] - 2):
            if g[i, w] < 0 and g[i, w + 1] < 0:
                out[i] = w
                break
    return out


def summarise(name, sp, T, reps, retro):
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
    """Pooled complete-case delays vs forward and backward pmfs."""
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


def main():
    rng = np.random.default_rng(20261027)
    allrows = []
    for name in SCEN:
        sp = scenario_setup(name)
        retroT = sp["L"] - 1 + 60
        cutoffs = [retroT, 90, 120] if name != "A" else [retroT, 70, 90]
        fwd = sp["fmat"] if name == "F" else None
        reps = [run_rep(name, sp, rng, cutoffs, fwd) for _ in range(NREP)]
        for T in cutoffs:
            allrows += summarise(name, sp, T, reps, T == retroT)
        print(f"done {name}", flush=True)
    keys = []
    for r in allrows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in allrows:
            w.writerow(
                {
                    k: (f"{v:.4f}" if isinstance(v, float) else v)
                    for k, v in r.items()
                }
            )
    with open(os.path.join(OUT, "analytic.txt"), "w") as fh:
        fh.write("\n".join(analytic()) + "\n")
    print("\n".join(analytic()))


if __name__ == "__main__":
    main()
