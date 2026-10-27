# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Make the figures used in the docs.

Writes SVGs to `docs/figures/`, a light and a `-dark` version of each.
The figures are built at docs build time and are not committed.

Every figure except the worked example uses expected values, not random
draws. They are computed here from the scenario settings in
`epidemic_dates` (`scenario_setup`, `F_DEFAULT`) and its allocation
functions (`hub_allocate`, `mean_shift_allocate`). The generative method
is EM back-projection with the true delay, run on the expected counts.
The worked example uses `impute_linelist` on the example line list.

Run from the repository root:

    uv run --with matplotlib python docs/figures/make_figures.py
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import epidemic_dates as ed

HERE = Path(__file__).parent
CSV = HERE.parent / "data" / "example-linelist.csv"

# Method colours, matching the interactive page.
COL = {
    "hub": "#2a78d6",
    "mean": "#eb6834",
    "fwd": "#1baf7a",
    "gen": "#eda100",
}
LABEL = {
    "hub": "Hub",
    "mean": "Mean shift",
    "fwd": "Forward draw",
    "gen": "Generative (EM)",
}
THEMES = {
    "light": {
        "truth": "#0b0b0b",
        "text": "#333333",
        "muted": "#6b6b6b",
        "grid": "#e2e2df",
        "bar": "#d6d6d2",
        "shade": "#000000",
    },
    "dark": {
        "truth": "#f2f2f0",
        "text": "#d6d6d4",
        "muted": "#9a9a97",
        "grid": "#3a3a38",
        "bar": "#4a4a47",
        "shade": "#ffffff",
    },
}

# Scenarios, all from `epidemic_dates.SCEN`.
STEADY = "A"  # r = 0.05 constant
TURN = "D"  # r = 0.06, then -0.04 from day 80
TURN_DAY = 80
REALTIME_T = 101  # three weeks after the turn
DELAY_T_STEADY = 100
DELAY_T_TURN = 90
TRUNC_T = 90
P_MISSING = 0.4  # every scenario used here
GI_MEAN, GI_SD = 15.0, 9.0


def style(theme):
    """Set matplotlib defaults for one theme."""
    th = THEMES[theme]
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "axes.edgecolor": th["muted"],
            "axes.labelcolor": th["text"],
            "axes.titlecolor": th["text"],
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "grid.color": th["grid"],
            "grid.linewidth": 0.6,
            "xtick.color": th["muted"],
            "ytick.color": th["muted"],
            "xtick.labelcolor": th["text"],
            "ytick.labelcolor": th["text"],
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.labelcolor": th["text"],
            "legend.fontsize": 9,
        }
    )
    return th


def save(fig, name, theme):
    """Save a figure as `name.svg` or `name-dark.svg`."""
    suffix = "" if theme == "light" else "-dark"
    fig.savefig(HERE / f"{name}{suffix}.svg", bbox_inches="tight")
    plt.close(fig)


# Expected values ---------------------------------------------------------


def cdf_at(f, x):
    """Delay CDF F(x), with F = 0 for x < 0 and 1 beyond the support."""
    cdf = np.cumsum(f)
    x = np.asarray(x)
    out = np.where(x < 0, 0.0, cdf[np.clip(x, 0, f.size - 1)])
    return np.where(x >= f.size, 1.0, out)


def expected(name, cutoff):
    """Compute expected data for a scenario cut at report day `cutoff`.

    Returns a dict of daily arrays on onset (or report) days 0..cutoff:
    the eventual onsets `truth`, the fraction reported by the cut-off
    `frac`, complete cases by onset `complete`, missing-onset cases by
    report `missing` and the pooled empirical delay `g`.
    """
    sp = ed.scenario_setup(name)
    f = ed.F_DEFAULT
    days = cutoff + 1
    onsets = np.zeros(days)
    n = min(days, sp["L"])
    onsets[:n] = sp["O"][:n]
    frac = cdf_at(f, cutoff - np.arange(days))
    complete = (1 - P_MISSING) * onsets * frac
    by_report = np.convolve(onsets, f)[:days]
    missing = P_MISSING * by_report
    cum = np.cumsum(onsets)
    d = np.arange(f.size)
    exposure = np.where(cutoff - d >= 0, cum[np.clip(cutoff - d, 0, None)], 0)
    g = f * exposure
    g /= g.sum()
    return {
        "truth": onsets,
        "frac": frac,
        "complete": complete,
        "missing": missing,
        "g": g,
        "f": f,
    }


def em_known_delay(complete, missing, f, cutoff, n_iter=300):
    """EM back-projection of expected counts with a known delay."""
    days = cutoff + 1
    frac = np.maximum(
        cdf_at(f, np.minimum(cutoff - np.arange(days), f.size)), 1e-9
    )
    lam = (complete / (1 - P_MISSING)) / np.maximum(frac, 0.05) + 0.1
    alloc = np.zeros(days)
    for _ in range(n_iter):
        denom = np.convolve(lam, f)[:days]
        ratio = np.where(denom > 0, missing / np.maximum(denom, 1e-300), 0)
        # alloc(s) = lam(s) * sum_d f(d) ratio(s + d)
        back = np.array(
            [np.dot(f[: days - s], ratio[s : s + f.size]) for s in range(days)]
        )
        alloc = lam * back
        lam = (complete + alloc) / frac
    return alloc


def allocations(ex, cutoff):
    """Missing-onset cases by onset day under each method."""
    g, f, m = ex["g"], ex["f"], ex["missing"]
    shift = ed.round_half_up(float(np.dot(np.arange(g.size), g)))
    return {
        "hub": ed.hub_allocate(m, g),
        "mean": ed.mean_shift_allocate(m, shift),
        "fwd": ed.hub_allocate(m, f),
        "gen": em_known_delay(ex["complete"], m, f, cutoff),
    }


def to_weeks(x):
    """Sum daily values to weeks, week 0 being days 0 to 6."""
    return np.bincount(np.arange(x.size) // 7, weights=x)


def rt_from_weeks(x):
    """Weekly Rt from weekly counts, as on the Methods page."""
    lx = np.log(np.maximum(x, 1e-9))
    r = np.empty_like(lx)
    r[1:-1] = (lx[2:] - lx[:-2]) / 14
    r[0] = (lx[1] - lx[0]) / 7
    r[-1] = (lx[-1] - lx[-2]) / 7
    k = (GI_MEAN / GI_SD) ** 2
    theta = GI_SD**2 / GI_MEAN
    return (1 + r * theta) ** k


def weekly_errors(name, cutoff, retro):
    """Weekly relative error and Rt error of each method."""
    ex = expected(name, cutoff)
    alloc = allocations(ex, cutoff)
    frac = ex["frac"]
    corr = np.where(frac >= 0.05, 1 / np.maximum(frac, 1e-9), 0.0)
    truth_w = to_weeks(ex["truth"])
    # complete weeks: case-weighted fraction reported at least 30%
    wfrac = to_weeks(ex["truth"] * frac) / np.maximum(truth_w, 1e-12)
    n_weeks = truth_w.size
    if retro:
        n_weeks = ed.SCEN[name].days // 7
    keep = (truth_w[:n_weeks] >= 10) & (wfrac[:n_weeks] >= 0.3)
    weeks = np.arange(n_weeks)[keep]
    lo, hi = weeks.min(), weeks.max() + 1
    tw = truth_w[lo:hi]
    out = {"week": np.arange(lo, hi), "rel": {}, "rt": {}}
    rt_truth = rt_from_weeks(tw)
    for m, a in alloc.items():
        est = to_weeks((ex["complete"] + a) * corr)[lo:hi]
        out["rel"][m] = est / tw - 1
        out["rt"][m] = rt_from_weeks(est) - rt_truth
    return out


# Figures -----------------------------------------------------------------


def fig_delays(theme):
    """Pooled, forward and backward delay for one report day."""
    th = style(theme)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    panels = [
        (STEADY, DELAY_T_STEADY, "Steady growth (r = 0.05 per day)"),
        (TURN, DELAY_T_TURN, "10 days after growth turns"),
    ]
    dmax = 40
    d = np.arange(dmax + 1)
    for ax, (name, t, title) in zip(axes, panels, strict=True):
        ex = expected(name, t)
        f, g = ex["f"], ex["g"]
        prior = ex["truth"][t - np.arange(f.size)]
        back = prior * f
        back /= back.sum()
        ax.plot(d, f[: dmax + 1], color=COL["fwd"], label="Forward delay")
        ax.plot(
            d, g[: dmax + 1], color=COL["hub"], label="Pooled empirical (hub)"
        )
        ax.plot(
            d,
            back[: dmax + 1],
            color=th["truth"],
            lw=1.2,
            ls=(0, (4, 2)),
            label="Correct backward delay",
        )
        ax.set_title(title)
        ax.set_xlabel("Delay from onset to report (days)")
        ax.set_xlim(0, dmax)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Probability")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "delays", theme)


def fig_placement(theme):
    """Where each method places a missing onset reported on one day."""
    th = style(theme)
    name, t = TURN, DELAY_T_TURN
    ex = expected(name, t)
    f, g = ex["f"], ex["g"]
    days = np.arange(t + 1)
    window = days >= t - 35
    prior = ex["truth"]
    lag = t - days
    inside = lag < f.size
    post = np.where(inside, prior * f[np.clip(lag, 0, f.size - 1)], 0.0)
    post /= post.sum()
    hub = np.where(inside, g[np.clip(lag, 0, f.size - 1)], 0.0)
    fwd = np.where(inside, f[np.clip(lag, 0, f.size - 1)], 0.0)
    shift = ed.round_half_up(float(np.dot(np.arange(g.size), g)))
    # the generative method's posterior at the EM fixed point
    lam = ex["complete"] + allocations(ex, t)["gen"]
    lam = lam / np.maximum(ex["frac"], 1e-9)
    gen = np.where(inside, lam * f[np.clip(lag, 0, f.size - 1)], 0.0)
    gen /= gen.sum()

    def mean_day(p):
        return float(np.dot(days, p) / p.sum())

    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = days[window]
    ax.axvline(TURN_DAY, color=th["muted"], lw=0.8, ls=":")
    ax.text(
        TURN_DAY - 0.4,
        0.97,
        "growth turns ",
        color=th["muted"],
        ha="right",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=9,
    )
    ax.plot(
        x,
        post[window],
        color=th["truth"],
        lw=1.2,
        zorder=5,
        label=f"Truth, mean day {mean_day(post):.1f}",
    )
    ax.plot(
        x,
        gen[window],
        color=COL["gen"],
        lw=4,
        alpha=0.55,
        label=f"{LABEL['gen']}, mean day {mean_day(gen):.1f}",
    )
    ax.plot(
        x,
        hub[window],
        color=COL["hub"],
        label=f"{LABEL['hub']}, mean day {mean_day(hub):.1f}",
    )
    ax.plot(
        x,
        fwd[window],
        color=COL["fwd"],
        label=f"{LABEL['fwd']}, mean day {mean_day(fwd):.1f}",
    )
    ax.vlines(
        t - shift,
        0,
        1,
        color=COL["mean"],
        lw=2,
        transform=ax.get_xaxis_transform(),
        label=f"{LABEL['mean']}, all on day {t - shift}",
    )
    ax.set_xlabel(f"Onset day (case reported on day {t})")
    ax.set_ylabel("Probability")
    ax.set_xlim(x.min(), t)
    ax.set_ylim(0, max(hub.max(), post.max(), fwd.max()) * 1.15)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    save(fig, "placement", theme)


def fig_changepoint(theme):
    """Weekly relative error and Rt error around a sharp turn."""
    th = style(theme)
    retro_t = ed.SCEN[TURN].days - 1 + ed.DMAX
    cols = [
        (weekly_errors(TURN, retro_t, True), "Retrospective, all reported"),
        (
            weekly_errors(TURN, REALTIME_T, False),
            f"Real time, cut at day {REALTIME_T}, corrected",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2), sharex=True, sharey="row")
    for j, (res, title) in enumerate(cols):
        for i, key in enumerate(["rel", "rt"]):
            ax = axes[i, j]
            ax.axhline(0, color=th["truth"], lw=0.8)
            ax.axvline(TURN_DAY / 7, color=th["muted"], lw=0.8, ls=":")
            for m in ["hub", "mean", "fwd", "gen"]:
                ls = (0, (3, 2)) if m == "fwd" else "-"
                ax.plot(
                    res["week"],
                    res[key][m],
                    color=COL[m],
                    ls=ls,
                    marker="o",
                    ms=3,
                    label=LABEL[m],
                )
            if i == 0:
                ax.set_title(title)
        if j == 1:
            for ax in axes[:, j]:
                ax.axvline(REALTIME_T / 7, color=th["muted"], lw=0.8)
            axes[0, j].text(
                REALTIME_T / 7,
                0.98,
                " cut-off",
                color=th["muted"],
                transform=axes[0, j].get_xaxis_transform(),
                fontsize=9,
                ha="left",
                va="top",
            )
    for ax in axes[0]:
        ax.text(
            TURN_DAY / 7,
            0.98,
            "turn ",
            color=th["muted"],
            ha="right",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
        )
    axes[0, 0].set_ylabel("Relative error\nin weekly onsets")
    axes[1, 0].set_ylabel("Error in Rt")
    for ax in axes[1]:
        ax.set_xlabel("Onset week")
    axes[0, 0].yaxis.set_major_formatter(
        mticker.PercentFormatter(1.0, decimals=0)
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02)
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "changepoint", theme)
    return cols


def fig_truncation(theme):
    """Show the F(T - s) correction near a cut-off."""
    th = style(theme)
    ex = expected(STEADY, TRUNC_T)
    alloc = allocations(ex, TRUNC_T)
    frac = ex["frac"]
    corr = np.where(frac >= 0.05, 1 / np.maximum(frac, 1e-9), 0.0)
    days = np.arange(TRUNC_T + 1)
    win = days >= TRUNC_T - 35
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, m in zip(axes, ["hub", "mean"], strict=True):
        raw = ex["complete"] + alloc[m]
        ax.plot(
            days[win],
            ex["truth"][win],
            color=th["truth"],
            lw=1.2,
            label="Eventual truth",
        )
        ax.plot(
            days[win],
            (raw * corr)[win],
            color=COL[m],
            label=r"Corrected by $F(T-s)$",
        )
        ax.plot(
            days[win],
            raw[win],
            color=COL[m],
            lw=1.2,
            ls=(0, (3, 2)),
            alpha=0.8,
            label="Reported by the cut-off",
        )
        ax.set_title(LABEL[m])
        ax.set_xlabel(f"Onset day (cut-off T = {TRUNC_T})")
        ax.set_xlim(days[win].min(), TRUNC_T)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Expected onsets per day")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "truncation", theme)


def read_linelist():
    """Read the example line list as day numbers."""
    with CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    def to_date(x):
        return dt.date.fromisoformat(x) if x else None

    onset_date = [to_date(r["onset_date"]) for r in rows]
    report_date = [to_date(r["report_date"]) for r in rows]
    day0 = min(d for d in onset_date + report_date if d is not None)
    onset = np.array(
        [np.nan if d is None else (d - day0).days for d in onset_date]
    )
    report = np.array([(d - day0).days for d in report_date])
    return day0, onset, report


def fig_linelist(theme):
    """Weekly counts from the example line list."""
    th = style(theme)
    day0, onset, report = read_linelist()
    res = ed.impute_linelist(onset, report, cutoff=int(report.max()))
    n_days = int(report.max()) + 1
    by_report = to_weeks(np.bincount(report.astype(int), minlength=n_days))
    complete = to_weeks(res["complete"])
    starts = [day0 + dt.timedelta(days=7 * w) for w in range(complete.size)]
    mid = [s + dt.timedelta(days=3) for s in starts]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(
        mid, by_report, width=6, color=th["bar"], label="Cases by report week"
    )
    ax.plot(
        mid,
        complete,
        color=th["truth"],
        lw=1.2,
        ls=(0, (3, 2)),
        label="Complete onsets only",
    )
    for m, key in [("hub", "hub"), ("mean", "mean"), ("gen", "em")]:
        ax.plot(
            mid,
            to_weeks(res["imputed"][key]),
            color=COL[m],
            marker="o",
            ms=3,
            label=f"Onsets after imputation: {LABEL[m]}",
        )
    cut = day0 + dt.timedelta(days=int(report.max()))
    ax.axvline(cut, color=th["muted"], lw=0.8)
    ax.text(
        cut,
        0.98,
        "cut-off ",
        color=th["muted"],
        ha="right",
        va="top",
        transform=ax.get_xaxis_transform(),
        fontsize=9,
    )
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=4, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax.set_ylabel("Cases per week")
    ax.set_xlabel("Week starting")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "linelist", theme)


def main():
    """Write every figure in both themes and print key numbers."""
    for theme in THEMES:
        fig_linelist(theme)
        fig_delays(theme)
        fig_placement(theme)
        cols = fig_changepoint(theme)
        fig_truncation(theme)
    for res, title in cols:
        print(title)
        for m in ["hub", "mean", "fwd", "gen"]:
            rel = np.abs(res["rel"][m]).max()
            rt = np.abs(res["rt"][m]).max()
            print(f"  {m:5} max |rel| {rel:.3f}  max |Rt err| {rt:.3f}")
    print(f"figures written to {HERE}")


if __name__ == "__main__":
    main()
