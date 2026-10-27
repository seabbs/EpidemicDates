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
        "note": "Reference outputs from the Python package. "
        "back-sampling, mean and forward must match exactly; "
        "em within 2% on weeks with at least 50 cases.",
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
    out.update(delay_change_cases())
    (HERE / "cross-language.json").write_text(json.dumps(out, indent=1))


def delay_change_cases() -> dict[str, object]:
    """Cases with a delay that changes over time.

    A scenario F line list (mean delay falling from 12 to 4 days), cut at
    day 100, imputed with per-period EM delays and with the true delay of
    each onset day supplied as a matrix.
    """
    rng = np.random.default_rng(7)
    setup = ed.scenario_setup("F")
    onset, report, missing = ed.simulate("F", setup, rng)
    cutoff, period = 100, 14
    keep = report <= cutoff
    onset, report, missing = onset[keep], report[keep], missing[keep]
    obs = np.where(missing, np.nan, onset).astype(float)
    base = {
        "onset": [
            None if m else int(o) for o, m in zip(onset, missing, strict=True)
        ],
        "report": [int(r) for r in report],
        "cutoff": cutoff,
        "max_delay": ed.DMAX,
    }
    res = ed.impute_linelist(obs, report, cutoff=cutoff, delay_period=period)
    delay = setup["fmat"][: cutoff + 1]
    plug = ed.impute_linelist(obs, report, cutoff=cutoff, delay=delay)

    def as_lists(imputed: dict[str, np.ndarray]) -> dict[str, list[float]]:
        return {k: np.asarray(v).tolist() for k, v in imputed.items()}

    return {
        "linelist_periods": {
            **base,
            "delay_period": period,
            "delay_shrinkage": ed.DELAY_SHRINKAGE,
            "imputed": as_lists(res["imputed"]),
            "em_delay_by_period": res["em_delay_by_period"].tolist(),
            "delay_period_start": res["delay_period_start"].tolist(),
        },
        "linelist_delay_matrix": {
            **base,
            "delay": delay.tolist(),
            "imputed": as_lists(plug["imputed"]),
        },
    }


if __name__ == "__main__":
    main()
