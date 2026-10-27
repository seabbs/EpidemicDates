# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Make the example line lists used in the docs.

The first line list simulates scenario `D` (growth at 0.06 per day that
turns to decline at -0.04 per day on day 80) with `epidemic_dates`, cuts
the data at report day 97, 17 days after the turn, and sets 25% of
onsets to missing at random.
Day 0 is 2026-05-03.

Because the data are simulated, the truth is known. The script also writes
`example-linelist-truth.csv` (the true onset of every case in the line
list) and `example-eventual-onsets.csv` (onsets by day of all cases,
including those reported after the cut-off), so the docs can show how
close each method gets.

The second line list, `example-linelist-changing-delays.csv`, uses the
onset curve of scenario `C` (growth that slows and turns to decline) with
a mean delay that falls from 12 to 5 days between onset days 40 and 100,
as a response scales up. 25% of onsets are missing at random. Its truth
files are `example-linelist-changing-delays-truth.csv`,
`example-changing-delays-eventual-onsets.csv` and
`example-changing-delays-true-delay.csv`, which holds the true delay pmf
of each onset day from day 0 of the line list to its cut-off.

Run from the repository root:

    uv run python docs/data/make_example.py
"""

import csv
import datetime as dt
from pathlib import Path

import numpy as np

import epidemic_dates as ed

HERE = Path(__file__).parent
DAY0 = dt.date(2026, 5, 3)

SEED = 17
SCENARIO = "D"
CUTOFF = 97
P_MISSING = 0.25
OUT = HERE / "example-linelist.csv"
TRUTH = HERE / "example-linelist-truth.csv"
EVENTUAL = HERE / "example-eventual-onsets.csv"

SEED_CHANGING = 78
CUTOFF_CHANGING = 120
P_MISSING_CHANGING = 0.25
CHANGING = HERE / "example-linelist-changing-delays.csv"


def date(day):
    """ISO date of simulation day `day`."""
    return (DAY0 + dt.timedelta(days=int(day))).isoformat()


def simulate_example(seed=SEED):
    """Simulate the first line list.

    Returns the onset and report days of the cases reported by the
    cut-off, sorted by report day, their missing-onset flags and the
    onsets by day of all cases.
    """
    rng = np.random.default_rng(seed)
    onset, report, _ = ed.simulate(SCENARIO, ed.scenario_setup(SCENARIO), rng)
    keep = report <= CUTOFF
    eventual = np.bincount(onset, minlength=CUTOFF + 1)[: CUTOFF + 1]
    onset, report = onset[keep], report[keep]
    order = np.lexsort((onset, report))
    onset, report = onset[order], report[order]
    missing = rng.random(onset.size) < P_MISSING
    return onset, report, missing, eventual


def changing_delay_pmfs(n_days):
    """Return the true mean delay and pmf of each onset day (second example)."""
    days = np.arange(n_days)
    means = np.interp(days, [0, 40, 100, n_days], [12, 12, 5, 5])
    return means, np.array([ed.gamma_pmf(m, 0.8 * m) for m in means])


def simulate_changing(seed=SEED_CHANGING):
    """Simulate the second line list, with delays that change over time.

    The cut-off is the last day at or before `CUTOFF_CHANGING` that makes
    the line list span whole weeks from its first date. Returns as for
    `simulate_example`, plus the cut-off, the first day and the true mean
    delay and pmf of each onset day.
    """
    rng = np.random.default_rng(seed)
    setup = ed.scenario_setup("C")
    means, fmat = changing_delay_pmfs(setup["L"])
    onset = np.repeat(np.arange(setup["L"]), rng.poisson(setup["O"]))
    cdf = np.cumsum(fmat, axis=1)
    delay = (rng.random(onset.size)[:, None] > cdf[onset]).sum(axis=1)
    report = onset + np.minimum(delay, ed.DMAX)
    missing = rng.random(onset.size) < P_MISSING_CHANGING
    first = int(min(onset[report <= CUTOFF_CHANGING].min(), report.min()))
    cutoff = CUTOFF_CHANGING - (CUTOFF_CHANGING - first + 1) % 7
    keep = report <= cutoff
    eventual = np.bincount(onset, minlength=cutoff + 1)[: cutoff + 1]
    onset, report, missing = onset[keep], report[keep], missing[keep]
    order = np.lexsort((onset, report))
    onset, report, missing = onset[order], report[order], missing[order]
    first = int(min(onset[~missing].min(), report.min()))
    return onset, report, missing, eventual, cutoff, first, means, fmat


def write_linelist(path, truth_path, eventual_path, data):
    """Write a line list, its true onsets and its eventual onsets."""
    onset, report, missing, eventual = data
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["case_id", "onset_date", "report_date"])
        cases = zip(onset, report, missing, strict=True)
        for i, (s, t, m) in enumerate(cases, 1):
            w.writerow([f"case{i:04d}", "" if m else date(s), date(t)])
    with truth_path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["case_id", "true_onset_date", "report_date"])
        for i, (s, t) in enumerate(zip(onset, report, strict=True), 1):
            w.writerow([f"case{i:04d}", date(s), date(t)])
    with eventual_path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["onset_date", "cases"])
        for day_, n in enumerate(eventual):
            w.writerow([date(day_), int(n)])
    print(f"{onset.size} cases, {missing.sum()} missing onsets -> {path}")


def main():
    """Write both example line lists and their truth files."""
    write_linelist(OUT, TRUTH, EVENTUAL, simulate_example())
    onset, report, missing, eventual, cutoff, first, means, fmat = (
        simulate_changing()
    )
    write_linelist(
        CHANGING,
        HERE / "example-linelist-changing-delays-truth.csv",
        HERE / "example-changing-delays-eventual-onsets.csv",
        (onset, report, missing, eventual),
    )
    with (HERE / "example-changing-delays-true-delay.csv").open(
        "w", newline=""
    ) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(
            ["onset_date", "mean_delay"] + [f"p{d}" for d in range(ed.DMAX + 1)]
        )
        for day_ in range(first, cutoff + 1):
            w.writerow(
                [date(day_), f"{means[day_]:.4f}"]
                + [f"{p:.10g}" for p in fmat[day_]]
            )


if __name__ == "__main__":
    main()
