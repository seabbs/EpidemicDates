# Copyright (c) 2026 Sam Abbott. MIT licence, see LICENSE.
"""Write reference outputs shared by the Python, JavaScript, R and Julia tests.

Run with: uv run python tests/fixtures/make_fixtures.py
"""

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

import epidemic_dates as ed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    """Write tests/fixtures/cross-language.json."""
    pmfs = []
    for mean, cv in [(8.0, 1.0), (6.6, 0.92), (14.0, 0.7)]:
        pmfs.append(
            {
                "mean": mean,
                "cv": cv,
                "dmax": ed.DMAX,
                "pmf": ed.gamma_pmf(mean, mean * cv).tolist(),
            }
        )
    with open(ROOT / "docs/data/example-linelist.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    dates = [date.fromisoformat(r["report_date"]) for r in rows] + [
        date.fromisoformat(r["onset_date"]) for r in rows if r["onset_date"]
    ]
    day0 = min(dates)
    report = [(date.fromisoformat(r["report_date"]) - day0).days for r in rows]
    onset = [
        (date.fromisoformat(r["onset_date"]) - day0).days
        if r["onset_date"]
        else None
        for r in rows
    ]
    cutoff = max(report)
    obs = np.array([np.nan if o is None else o for o in onset], dtype=float)
    res = ed.impute_linelist(obs, np.array(report), cutoff=cutoff)
    out = {
        "note": "Reference outputs from the Python package. hub and mean "
        "must match exactly; em within 2% on weeks with at least 50 cases.",
        "delay_pmfs": pmfs,
        "linelist": {
            "onset": onset,
            "report": report,
            "cutoff": cutoff,
            "max_delay": ed.DMAX,
            "imputed": {
                k: np.asarray(v).tolist() for k, v in res["imputed"].items()
            },
            "empirical_delay": np.asarray(res["empirical_delay"]).tolist(),
        },
    }
    (HERE / "cross-language.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
