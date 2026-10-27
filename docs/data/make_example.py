# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Make the example line list used in the docs.

Simulates scenario `C` (growth that slows and turns to decline) with
`epidemic_dates`, keeps a random 12% of cases, cuts the data at report
day 120, and sets 15% of onsets to missing at random.
Day 0 is 2026-05-03.

Run from the repository root:

    uv run python docs/data/make_example.py
"""

import csv
import datetime as dt
from pathlib import Path

import numpy as np

import epidemic_dates as oi

SEED = 20261027
DAY0 = dt.date(2026, 5, 3)
KEEP = 0.12
CUTOFF = 120
P_MISSING = 0.15
OUT = Path(__file__).parent / "example-linelist.csv"


def main():
    rng = np.random.default_rng(SEED)
    onset, report, _ = oi.simulate("C", oi.scenario_setup("C"), rng)
    keep = (rng.random(onset.size) < KEEP) & (report <= CUTOFF)
    onset, report = onset[keep], report[keep]
    order = np.lexsort((onset, report))
    onset, report = onset[order], report[order]
    missing = rng.random(onset.size) < P_MISSING

    def date(day):
        return (DAY0 + dt.timedelta(days=int(day))).isoformat()

    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["case_id", "onset_date", "report_date"])
        for i, (s, t, m) in enumerate(zip(onset, report, missing), 1):
            w.writerow([f"case{i:04d}", "" if m else date(s), date(t)])
    print(f"{onset.size} cases, {missing.sum()} missing onsets -> {OUT}")


if __name__ == "__main__":
    main()
