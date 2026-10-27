# Using it on a real line list

This page works through imputing missing onsets in a line list, step by step, in Python and JavaScript.
Pick a language in any tab and the other tabs follow.

The functions are `impute_linelist` in the Python package ([source](https://github.com/seabbs/epidemic-imputation-demonstration/tree/main/src/epidemic_dates)) and `imputeLinelist` in the JavaScript module ([source](https://github.com/seabbs/epidemic-imputation-demonstration/blob/main/epidemic-dates.js)).
Both return daily onset counts after imputation under each method.
Read the [Assumptions](assumptions.md) before relying on the output.

## The example data

[Download `example-linelist.csv`](data/example-linelist.csv){ .md-button }

The file has 437 cases, one row per case, and three columns:

| Column | Content |
|---|---|
| `case_id` | Case identifier. |
| `onset_date` | Symptom onset date (ISO format), empty where missing. |
| `report_date` | Report date (ISO format), always present. |

```text
case_id,onset_date,report_date
case0001,2026-05-08,2026-05-09
case0002,2026-05-13,2026-05-19
case0003,2026-05-10,2026-05-23
...
case0435,,2026-08-31
```

76 cases (17%) have no onset date.
The data are a sample from the Python package's scenario `C`, in which growth slows and turns to decline.
The data are cut at the last report date, 2026-08-31, so recent onsets are still being reported.
The script that made the file is [`docs/data/make_example.py`](https://github.com/seabbs/epidemic-imputation-demonstration/blob/main/docs/data/make_example.py).

To use your own data, replace the file name and column names.
Run the code below from the repository root, or change the paths.

## Step 1: read the line list

=== "Python"

    The `csv` module is enough.
    pandas works as well.

    ```python
    import csv
    import datetime as dt

    import numpy as np

    import epidemic_dates as oi

    with open("docs/data/example-linelist.csv") as fh:
        rows = list(csv.DictReader(fh))
    ```

=== "JavaScript"

    The file is simple enough to parse without a CSV library.
    Use one if your data have quoted fields.

    ```js
    const fs = require("node:fs");
    const { imputeLinelist } = require("./epidemic-dates.js");

    const [header, ...lines] = fs
      .readFileSync("docs/data/example-linelist.csv", "utf8")
      .trim()
      .split(/\r?\n/);
    const cols = header.split(",");
    const rows = lines.map((line) => {
      const v = line.split(",");
      return Object.fromEntries(cols.map((c, i) => [c, v[i]]));
    });
    ```

## Step 2: convert dates to day numbers

Both functions take whole-number days.
Count days from the earliest date in either column, so no known onset falls before day 0.
Missing onsets become `NaN` in Python and `null` in JavaScript.

=== "Python"

    ```python
    def to_date(x):
        return dt.date.fromisoformat(x) if x else None

    onset_date = [to_date(r["onset_date"]) for r in rows]
    report_date = [to_date(r["report_date"]) for r in rows]
    day0 = min(d for d in onset_date + report_date if d is not None)
    onset = np.array(
        [np.nan if d is None else (d - day0).days for d in onset_date]
    )
    report = np.array([(d - day0).days for d in report_date])
    print(f"{len(rows)} cases, {np.isnan(onset).sum()} missing onsets")
    print(f"day 0 is {day0}, cut-off is day {report.max()}")
    ```

=== "JavaScript"

    ```js
    const DAY = 24 * 60 * 60 * 1000;
    const toTime = (x) => (x ? Date.parse(x) : null);
    const onsetTime = rows.map((r) => toTime(r.onset_date));
    const reportTime = rows.map((r) => toTime(r.report_date));
    const t0 = Math.min(...onsetTime.filter((x) => x !== null), ...reportTime);
    const onset = onsetTime.map((x) => (x === null ? null : (x - t0) / DAY));
    const report = reportTime.map((x) => (x - t0) / DAY);
    const cutoff = Math.max(...report);
    const nMissing = onset.filter((x) => x === null).length;
    console.log(`${rows.length} cases, ${nMissing} missing onsets`);
    const day0 = new Date(t0).toISOString().slice(0, 10);
    console.log(`day 0 is ${day0}, cut-off is day ${cutoff}`);
    ```

```text
437 cases, 76 missing onsets
day 0 is 2026-05-08, cut-off is day 115
```

## Step 3: impute

Set the cut-off to the last report day in the data.
Cases with an onset after their report, or a delay longer than the maximum, are dropped and counted in `dropped`.

=== "Python"

    ```python
    res = oi.impute_linelist(onset, report, cutoff=report.max())
    print(f"dropped: {res['dropped']}")
    ```

    `res["imputed"]` holds daily onset counts for `hub`, `mean` and `em`.
    `res["complete"]` holds the cases with a known onset.
    `res["empirical_delay"]` is the pooled delay pmf of complete cases.

=== "JavaScript"

    ```js
    const res = imputeLinelist({ onset, report, cutoff });
    console.log(`dropped: ${res.dropped}`);
    ```

    `res.imputed` holds daily onset counts for `hub`, `mean` and `gen`.
    `res.imputed.fwd` is `null` unless you pass a forward delay as `delay`.
    `res.complete` holds the cases with a known onset.
    `res.generativeDelay` is the delay estimated by EM, allowing for right truncation.

```text
dropped: 0
```

Without a `delay`, the EM method estimates one static delay from the data.
Pass a delay pmf indexed by day to hold it fixed instead.
In JavaScript that also fills in the forward draw.

## Step 4: sum to onset weeks

Day `i` of each output is onset day `i`.
Daily counts are noisy, so sum them to weeks before comparing methods.
Week 0 is days 0 to 6.

=== "Python"

    ```python
    def to_weeks(daily):
        weeks = np.arange(len(daily)) // 7
        return np.bincount(weeks, weights=daily)

    complete = to_weeks(res["complete"])
    weekly = {m: to_weeks(res["imputed"][m]) for m in ["hub", "mean", "em"]}
    ```

=== "JavaScript"

    ```js
    const toWeeks = (daily) => {
      const weeks = new Array(Math.ceil(daily.length / 7)).fill(0);
      daily.forEach((x, i) => (weeks[Math.floor(i / 7)] += x));
      return weeks;
    };
    const complete = toWeeks(res.complete);
    const methods = ["hub", "mean", "gen"];
    const weekly = Object.fromEntries(
      methods.map((m) => [m, toWeeks(res.imputed[m])])
    );
    ```

## Step 5: show weekly counts by method

=== "Python"

    ```python
    print(f"{'week from':<11}{'complete':>9}{'hub':>8}{'mean':>8}{'em':>8}")
    for w in range(len(complete)):
        start = day0 + dt.timedelta(days=7 * w)
        print(
            f"{start.isoformat():<11}{complete[w]:>9.0f}"
            + "".join(f"{weekly[m][w]:>8.1f}" for m in ["hub", "mean", "em"])
        )
    ```

    ```text
    week from   complete     hub    mean      em
    2026-05-08         4     4.2     4.0     4.2
    2026-05-15         0     0.5     0.0     0.0
    2026-05-22         2     3.2     5.0     2.5
    2026-05-29         5     6.8     6.0     7.8
    2026-06-05         6     7.2     6.0     7.0
    2026-06-12         9    10.6    11.0    10.1
    2026-06-19        16    19.8    19.0    20.0
    2026-06-26        23    27.3    29.0    27.3
    2026-07-03        29    35.5    34.0    35.9
    2026-07-10        39    43.4    43.0    43.7
    2026-07-17        34    41.9    43.0    41.0
    2026-07-24        54    63.2    62.0    65.7
    2026-07-31        44    54.0    55.0    54.2
    2026-08-07        40    50.5    53.0    50.1
    2026-08-14        30    38.6    38.0    37.5
    2026-08-21        22    25.4    25.0    24.8
    2026-08-28         4     5.0     4.0     5.1
    ```

=== "JavaScript"

    ```js
    console.table(
      complete.map((c, w) => ({
        "week from": new Date(t0 + 7 * w * DAY).toISOString().slice(0, 10),
        complete: c,
        ...Object.fromEntries(methods.map((m) => [m, +weekly[m][w].toFixed(1)]))
      }))
    );
    ```

    ```text
    ┌─────────┬──────────────┬──────────┬──────┬──────┬──────┐
    │ (index) │ week from    │ complete │ hub  │ mean │ gen  │
    ├─────────┼──────────────┼──────────┼──────┼──────┼──────┤
    │ 0       │ '2026-05-08' │ 4        │ 4.2  │ 4    │ 4.2  │
    │ 1       │ '2026-05-15' │ 0        │ 0.5  │ 0    │ 0    │
    │ 2       │ '2026-05-22' │ 2        │ 3.2  │ 5    │ 2.5  │
    │ 3       │ '2026-05-29' │ 5        │ 6.8  │ 6    │ 7.8  │
    │ 4       │ '2026-06-05' │ 6        │ 7.2  │ 6    │ 7    │
    │ 5       │ '2026-06-12' │ 9        │ 10.6 │ 11   │ 10.1 │
    │ 6       │ '2026-06-19' │ 16       │ 19.8 │ 19   │ 20   │
    │ 7       │ '2026-06-26' │ 23       │ 27.3 │ 29   │ 27.3 │
    │ 8       │ '2026-07-03' │ 29       │ 35.5 │ 34   │ 35.9 │
    │ 9       │ '2026-07-10' │ 39       │ 43.4 │ 43   │ 43.7 │
    │ 10      │ '2026-07-17' │ 34       │ 41.9 │ 43   │ 41   │
    │ 11      │ '2026-07-24' │ 54       │ 63.2 │ 62   │ 65.7 │
    │ 12      │ '2026-07-31' │ 44       │ 54   │ 55   │ 54.2 │
    │ 13      │ '2026-08-07' │ 40       │ 50.5 │ 53   │ 50.1 │
    │ 14      │ '2026-08-14' │ 30       │ 38.6 │ 38   │ 37.5 │
    │ 15      │ '2026-08-21' │ 22       │ 25.4 │ 25   │ 24.8 │
    │ 16      │ '2026-08-28' │ 4        │ 5    │ 4    │ 5.1  │
    └─────────┴──────────────┴──────────┴──────┴──────┴──────┘
    ```

`em` in Python and `gen` in JavaScript are the same method and give the same numbers here.

## Reading the output

- Every method adds the 76 missing-onset cases back to the onset curve, so each column sums to about 437.
- The hub method and EM back-projection agree closely in most weeks here, because the delay is short and growth changes slowly.
  On the [interactive page](https://seabbs.github.io/epidemic-imputation-demonstration/) you can see when they part.
- The mean shift moves every missing case by the same number of days, so it is lumpier.
- The last weeks are incomplete under every method.
  Cases with onsets in late August are still being reported on 31 August.
  Correct for right truncation, or leave these weeks out, before estimating growth.
- Outputs are expected allocations, not random draws, so they need not be whole numbers.

## Before you rely on it

- Check that onsets are missing at random given the report day.
- Check whether delays change over the data vintage.
- If you fit your own delay to pass as `delay`, account for double interval censoring and right truncation.

The [Assumptions](assumptions.md) page has the details.
For production use, a model that treats missing onsets as part of the observation process is better.
An example is [epinowcast](https://package.epinowcast.org/articles/model.html#accounting-for-reported-cases-with-a-missing-reference-date).
