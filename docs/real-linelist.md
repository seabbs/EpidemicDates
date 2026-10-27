# Using it on a real line list

This page works through imputing missing onsets in a line list, step by step, in Python, JavaScript, R and Julia.
Pick a language in any tab and the other tabs follow.

The functions are:

- `impute_linelist` in the Python package ([source](https://github.com/seabbs/epidemic-imputation-demonstration/tree/main/src/epidemic_dates), [guide](python/index.md));
- `imputeLinelist` in the JavaScript module ([source](https://github.com/seabbs/epidemic-imputation-demonstration/blob/main/epidemic-dates.js), [guide](javascript/index.md));
- `impute_linelist` in the R package `epidemicdates` ([source](https://github.com/seabbs/epidemic-imputation-demonstration/tree/main/r), [guide](r/index.md));
- `impute_linelist` in the Julia package `EpidemicDates.jl` ([source](https://github.com/seabbs/epidemic-imputation-demonstration/tree/main/julia), [guide](julia/index.md)).

Each returns daily onset counts after imputation under each method.
Step 6 nowcasts the recent weeks with uncertainty, using `nowcast_linelist` in Python and `nowcastLinelist` in JavaScript.
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

    import epidemic_dates as ed

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

=== "R"

    Base R's `read.csv` is enough.
    Empty onset dates become `NA`.

    ```r
    library(epidemicdates)

    rows <- read.csv("docs/data/example-linelist.csv", na.strings = "")
    ```

=== "Julia"

    The file has no quoted fields, so splitting on commas is enough.
    Use CSV.jl for real data.

    ```julia
    using Dates
    using EpidemicDates

    rows = split.(readlines("docs/data/example-linelist.csv")[2:end], ",")
    ```

## Step 2: convert dates to day numbers

The functions take whole-number days.
Count days from the earliest date in either column, so no known onset falls before day 0.
Missing onsets become `NaN` in Python, `null` in JavaScript, `NA` in R and `missing` in Julia.
The R and Julia functions also take dates directly and count days the same way.

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

=== "R"

    ```r
    onset_date <- as.Date(rows$onset_date)
    report_date <- as.Date(rows$report_date)
    day0 <- min(onset_date, report_date, na.rm = TRUE)
    onset <- as.numeric(onset_date - day0)
    report <- as.numeric(report_date - day0)
    cat(sprintf("%d cases, %d missing onsets\n", nrow(rows), sum(is.na(onset))))
    cat(sprintf("day 0 is %s, cut-off is day %d\n", day0, max(report)))
    ```

=== "Julia"

    ```julia
    onset_date = [isempty(r[2]) ? missing : Date(r[2]) for r in rows]
    report_date = [Date(r[3]) for r in rows]
    day0 = min(minimum(skipmissing(onset_date)), minimum(report_date))
    onset = [ismissing(d) ? missing : Dates.value(d - day0) for d in onset_date]
    report = [Dates.value(d - day0) for d in report_date]
    println("$(length(rows)) cases, $(count(ismissing, onset)) missing onsets")
    println("day 0 is $day0, cut-off is day $(maximum(report))")
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
    res = ed.impute_linelist(onset, report, cutoff=report.max())
    print(f"dropped: {res['dropped']}")
    ```

    `res["imputed"]` holds daily onset counts for `backsampling`, `mean` and `em`.
    `res["complete"]` holds the cases with a known onset.
    `res["empirical_delay"]` is the pooled delay pmf of complete cases.

=== "JavaScript"

    ```js
    const res = imputeLinelist({ onset, report, cutoff });
    console.log(`dropped: ${res.dropped}`);
    ```

    `res.imputed` holds daily onset counts for `backsampling`, `mean` and `gen`.
    `res.imputed.fwd` is `null` unless you pass a forward delay as `delay`.
    `res.complete` holds the cases with a known onset.
    `res.generativeDelay` is the delay estimated by EM, allowing for right truncation.

=== "R"

    ```r
    res <- impute_linelist(onset, report, cutoff = max(report))
    cat(sprintf("dropped: %d\n", res$dropped))
    ```

    `res$imputed` holds daily onset counts for `backsampling`, `mean` and `em`.
    `res$complete` holds the cases with a known onset.
    `res$em_delay` is the delay estimated by EM, allowing for right truncation.

=== "Julia"

    ```julia
    res = impute_linelist(onset, report; cutoff = maximum(report))
    println("dropped: $(res.dropped)")
    ```

    `res.imputed` holds daily onset counts for `backsampling`, `mean` and `em`.
    `res.complete` holds the cases with a known onset.
    `res.em_delay` is the delay estimated by EM, allowing for right truncation.

```text
dropped: 0
```

The EM method estimates one static delay from the data.
In JavaScript, pass a delay pmf indexed by day as `delay` to hold it fixed instead.
That also fills in the forward draw.
In R and Julia, a `delay` fills in the forward draw, and EM still estimates its own.

## Step 4: sum to onset weeks

Day `i` of each output is onset day `i`.
Outputs are expected allocations, not random draws, so they need not be whole numbers.
Daily counts are noisy, so sum them to weeks before comparing methods.
Week 0 is days 0 to 6.

=== "Python"

    ```python
    def to_weeks(daily):
        weeks = np.arange(len(daily)) // 7
        return np.bincount(weeks, weights=daily)

    complete = to_weeks(res["complete"])
    weekly = {m: to_weeks(res["imputed"][m]) for m in ["backsampling", "mean", "em"]}
    ```

=== "JavaScript"

    ```js
    const toWeeks = (daily) => {
      const weeks = new Array(Math.ceil(daily.length / 7)).fill(0);
      daily.forEach((x, i) => (weeks[Math.floor(i / 7)] += x));
      return weeks;
    };
    const complete = toWeeks(res.complete);
    const methods = ["backsampling", "mean", "gen"];
    const weekly = Object.fromEntries(
      methods.map((m) => [m, toWeeks(res.imputed[m])])
    );
    ```

=== "R"

    `weekly()` sums days into weeks starting on day 0.

    ```r
    complete <- weekly(res$complete)
    weekly_counts <- lapply(res$imputed, weekly)
    ```

=== "Julia"

    `weekly` sums days into weeks starting on day 0.

    ```julia
    complete = weekly(res.complete)
    weeks = Dict(m => weekly(res.imputed[m]) for m in (:backsampling, :mean, :em))
    ```

## Step 5: show weekly counts by method

=== "Python"

    ```python
    print(f"{'week from':<11}{'complete':>9}{'backsampling':>13}{'mean':>8}{'em':>8}")
    for w in range(len(complete)):
        start = day0 + dt.timedelta(days=7 * w)
        print(
            f"{start.isoformat():<11}{complete[w]:>9.0f}"
            f"{weekly['backsampling'][w]:>13.1f}"
            + "".join(f"{weekly[m][w]:>8.1f}" for m in ["mean", "em"])
        )
    ```

    ```text
    week from   complete backsampling    mean      em
    2026-05-08         4          4.2     4.0     4.2
    2026-05-15         0          0.5     0.0     0.0
    2026-05-22         2          3.2     5.0     2.5
    2026-05-29         5          6.8     6.0     7.8
    2026-06-05         6          7.2     6.0     7.0
    2026-06-12         9         10.6    11.0    10.1
    2026-06-19        16         19.8    19.0    20.0
    2026-06-26        23         27.3    29.0    27.3
    2026-07-03        29         35.5    34.0    35.9
    2026-07-10        39         43.4    43.0    43.7
    2026-07-17        34         41.9    43.0    41.0
    2026-07-24        54         63.2    62.0    65.7
    2026-07-31        44         54.0    55.0    54.2
    2026-08-07        40         50.5    53.0    50.1
    2026-08-14        30         38.6    38.0    37.5
    2026-08-21        22         25.4    25.0    24.8
    2026-08-28         4          5.0     4.0     5.1
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
    ┌─────────┬──────────────┬──────────┬──────────────┬──────┬──────┐
    │ (index) │ week from    │ complete │ backsampling │ mean │ gen  │
    ├─────────┼──────────────┼──────────┼──────────────┼──────┼──────┤
    │ 0       │ '2026-05-08' │ 4        │ 4.2          │ 4    │ 4.2  │
    │ 1       │ '2026-05-15' │ 0        │ 0.5          │ 0    │ 0    │
    │ 2       │ '2026-05-22' │ 2        │ 3.2          │ 5    │ 2.5  │
    │ 3       │ '2026-05-29' │ 5        │ 6.8          │ 6    │ 7.8  │
    │ 4       │ '2026-06-05' │ 6        │ 7.2          │ 6    │ 7    │
    │ 5       │ '2026-06-12' │ 9        │ 10.6         │ 11   │ 10.1 │
    │ 6       │ '2026-06-19' │ 16       │ 19.8         │ 19   │ 20   │
    │ 7       │ '2026-06-26' │ 23       │ 27.3         │ 29   │ 27.3 │
    │ 8       │ '2026-07-03' │ 29       │ 35.5         │ 34   │ 35.9 │
    │ 9       │ '2026-07-10' │ 39       │ 43.4         │ 43   │ 43.7 │
    │ 10      │ '2026-07-17' │ 34       │ 41.9         │ 43   │ 41   │
    │ 11      │ '2026-07-24' │ 54       │ 63.2         │ 62   │ 65.7 │
    │ 12      │ '2026-07-31' │ 44       │ 54           │ 55   │ 54.2 │
    │ 13      │ '2026-08-07' │ 40       │ 50.5         │ 53   │ 50.1 │
    │ 14      │ '2026-08-14' │ 30       │ 38.6         │ 38   │ 37.5 │
    │ 15      │ '2026-08-21' │ 22       │ 25.4         │ 25   │ 24.8 │
    │ 16      │ '2026-08-28' │ 4        │ 5            │ 4    │ 5.1  │
    └─────────┴──────────────┴──────────┴──────────────┴──────┴──────┘
    ```

=== "R"

    ```r
    data.frame(
      week_from = day0 + 7 * (seq_along(complete) - 1),
      complete = complete,
      lapply(weekly_counts, round, 1)
    )
    ```

    ```text
        week_from complete backsampling mean   em
    1  2026-05-08        4          4.2    4  4.2
    2  2026-05-15        0          0.5    0  0.0
    3  2026-05-22        2          3.2    5  2.5
    4  2026-05-29        5          6.8    6  7.8
    5  2026-06-05        6          7.2    6  7.0
    6  2026-06-12        9         10.6   11 10.1
    7  2026-06-19       16         19.8   19 20.0
    8  2026-06-26       23         27.3   29 27.3
    9  2026-07-03       29         35.5   34 35.9
    10 2026-07-10       39         43.4   43 43.7
    11 2026-07-17       34         41.9   43 41.0
    12 2026-07-24       54         63.2   62 65.7
    13 2026-07-31       44         54.0   55 54.2
    14 2026-08-07       40         50.5   53 50.1
    15 2026-08-14       30         38.6   38 37.5
    16 2026-08-21       22         25.4   25 24.8
    17 2026-08-28        4          5.0    4  5.1
    ```

=== "Julia"

    ```julia
    println(rpad("week from", 11), lpad("complete", 9), lpad("backsampling", 13),
        lpad("mean", 8), lpad("em", 8))
    for w in eachindex(complete)
        start = day0 + Day(7 * (w - 1))
        println(rpad(string(start), 11), lpad(round(Int, complete[w]), 9),
            lpad(round(weeks[:backsampling][w]; digits = 1), 13),
            join(lpad(round(weeks[m][w]; digits = 1), 8) for m in (:mean, :em)))
    end
    ```

    ```text
    week from   complete backsampling    mean      em
    2026-05-08         4          4.2     4.0     4.2
    2026-05-15         0          0.5     0.0     0.0
    2026-05-22         2          3.2     5.0     2.5
    2026-05-29         5          6.8     6.0     7.8
    2026-06-05         6          7.2     6.0     7.0
    2026-06-12         9         10.6    11.0    10.1
    2026-06-19        16         19.8    19.0    20.0
    2026-06-26        23         27.3    29.0    27.3
    2026-07-03        29         35.5    34.0    35.9
    2026-07-10        39         43.4    43.0    43.7
    2026-07-17        34         41.9    43.0    41.0
    2026-07-24        54         63.2    62.0    65.7
    2026-07-31        44         54.0    55.0    54.2
    2026-08-07        40         50.5    53.0    50.1
    2026-08-14        30         38.6    38.0    37.5
    2026-08-21        22         25.4    25.0    24.8
    2026-08-28         4          5.0     4.0     5.1
    ```

`em` in Python, R and Julia and `gen` in JavaScript are the same method and give the same numbers here.
Each column adds the 76 missing-onset cases back, so it sums to about 437.
The back-sampling method and EM back-projection agree closely in most weeks here, because the delay is short and growth changes slowly.
On the [interactive page](https://seabbs.github.io/epidemic-imputation-demonstration/) you can see when they part.
The mean shift moves every missing case by the same number of days, so it is lumpier.

<figure class="doc-figure" markdown="span">
  ![Weekly cases in the example line list from May to early September 2026. Grey bars show cases by report week. Lines show onsets with a known date and onsets after imputation by the back-sampling, mean shift and EM methods. The three imputed curves lie close together, peak in the week of 24 July and sit above the known-onset curve.](figures/linelist.svg#only-light) ![Weekly cases in the example line list from May to early September 2026. Grey bars show cases by report week. Lines show onsets with a known date and onsets after imputation by the back-sampling, mean shift and EM methods. The three imputed curves lie close together, peak in the week of 24 July and sit above the known-onset curve.](figures/linelist-dark.svg#only-dark)
  <figcaption markdown="1">
    Weekly counts from the example line list.
    Grey bars are cases by report week.
    The dashed line counts cases with a known onset.
    Coloured lines add the 76 missing-onset cases back by each method.
    The data are cut on 31 August, so the last weeks are incomplete under every method.
    Cases with onsets in late August are still being reported.
    Step 6 corrects for this.
  </figcaption>
</figure>

## Step 6: nowcast the recent weeks

The imputed counts above miss cases with recent onsets that are not yet reported.
A nowcast corrects for this and says how uncertain the correction is.
It imputes the missing onsets as before, here with EM back-projection.
It then divides each onset day by the estimated fraction of its cases reported by the cut-off, $F(T - s)$.
It repeats this with random draws to give quantiles.
The [Methods](methods.md#nowcast-with-uncertainty) page has the maths.
The nowcast is currently in Python and JavaScript only.

=== "Python"

    ```python
    nc = ed.nowcast_linelist(
        onset, report, cutoff=report.max(), method="em", seed=1, origin=day0
    )
    last = sorted({r["week_start"] for r in nc["weekly_quantiles"]})[-3:]
    print(f"{'week_start':<12}{'target_end_date':<17}{'quantile':>8}{'value':>7}")
    for r in nc["weekly_quantiles"]:
        if r["week_start"] in last:
            print(
                f"{r['week_start']:<12}{r['target_end_date']:<17}"
                f"{r['quantile']:>8}{r['value']:>7.1f}"
            )
    print(f"weeks without an estimate: {np.isnan(nc['point_weekly']).sum()}")
    ed.write_nowcast_csv(nc, "nowcast.csv")
    ```

    ```text
    week_start  target_end_date  quantile  value
    2026-08-04  2026-08-10          0.025   53.0
    2026-08-04  2026-08-10            0.1   55.0
    2026-08-04  2026-08-10           0.25   57.0
    2026-08-04  2026-08-10            0.5   59.0
    2026-08-04  2026-08-10           0.75   61.0
    2026-08-04  2026-08-10            0.9   63.0
    2026-08-04  2026-08-10          0.975   66.0
    2026-08-11  2026-08-17          0.025   44.0
    2026-08-11  2026-08-17            0.1   47.0
    2026-08-11  2026-08-17           0.25   49.0
    2026-08-11  2026-08-17            0.5   52.0
    2026-08-11  2026-08-17           0.75   54.0
    2026-08-11  2026-08-17            0.9   57.0
    2026-08-11  2026-08-17          0.975   60.0
    2026-08-18  2026-08-24          0.025   39.0
    2026-08-18  2026-08-24            0.1   42.0
    2026-08-18  2026-08-24           0.25   44.0
    2026-08-18  2026-08-24            0.5   48.0
    2026-08-18  2026-08-24           0.75   51.0
    2026-08-18  2026-08-24            0.9   55.0
    2026-08-18  2026-08-24          0.975   60.0
    weeks without an estimate: 1
    ```

    `write_nowcast_csv` writes the same weekly table to a CSV file.
    Pass `by="day"` for the daily table.
    `nc["weekly_samples"]` and `nc["samples"]` hold the samples behind the quantiles.

=== "JavaScript"

    ```js
    const { nowcastLinelist } = require("./epidemic-dates.js");
    const nc = nowcastLinelist({
      onset,
      report,
      cutoff,
      method: "gen",
      seed: 1,
      origin: day0
    });
    const last = [...new Set(nc.weeklyQuantiles.map((r) => r.week_start))].slice(-3);
    console.table(
      nc.weeklyQuantiles
        .filter((r) => last.includes(r.week_start))
        .map((r) => ({ ...r, value: +r.value.toFixed(1) }))
    );
    const noEstimate = nc.pointWeekly.filter(Number.isNaN).length;
    console.log(`weeks without an estimate: ${noEstimate}`);
    ```

    ```text
    ┌─────────┬──────────────┬─────────────────┬──────────┬───────┐
    │ (index) │ week_start   │ target_end_date │ quantile │ value │
    ├─────────┼──────────────┼─────────────────┼──────────┼───────┤
    │ 0       │ '2026-08-04' │ '2026-08-10'    │ 0.025    │ 53    │
    │ 1       │ '2026-08-04' │ '2026-08-10'    │ 0.1      │ 55    │
    │ 2       │ '2026-08-04' │ '2026-08-10'    │ 0.25     │ 56    │
    │ 3       │ '2026-08-04' │ '2026-08-10'    │ 0.5      │ 59    │
    │ 4       │ '2026-08-04' │ '2026-08-10'    │ 0.75     │ 61    │
    │ 5       │ '2026-08-04' │ '2026-08-10'    │ 0.9      │ 63    │
    │ 6       │ '2026-08-04' │ '2026-08-10'    │ 0.975    │ 65    │
    │ 7       │ '2026-08-11' │ '2026-08-17'    │ 0.025    │ 45    │
    │ 8       │ '2026-08-11' │ '2026-08-17'    │ 0.1      │ 47    │
    │ 9       │ '2026-08-11' │ '2026-08-17'    │ 0.25     │ 49    │
    │ 10      │ '2026-08-11' │ '2026-08-17'    │ 0.5      │ 52    │
    │ 11      │ '2026-08-11' │ '2026-08-17'    │ 0.75     │ 54    │
    │ 12      │ '2026-08-11' │ '2026-08-17'    │ 0.9      │ 58    │
    │ 13      │ '2026-08-11' │ '2026-08-17'    │ 0.975    │ 60    │
    │ 14      │ '2026-08-18' │ '2026-08-24'    │ 0.025    │ 39    │
    │ 15      │ '2026-08-18' │ '2026-08-24'    │ 0.1      │ 42    │
    │ 16      │ '2026-08-18' │ '2026-08-24'    │ 0.25     │ 45    │
    │ 17      │ '2026-08-18' │ '2026-08-24'    │ 0.5      │ 48    │
    │ 18      │ '2026-08-18' │ '2026-08-24'    │ 0.75     │ 51    │
    │ 19      │ '2026-08-18' │ '2026-08-24'    │ 0.9      │ 55    │
    │ 20      │ '2026-08-18' │ '2026-08-24'    │ 0.975    │ 58    │
    └─────────┴──────────────┴─────────────────┴──────────┴───────┘
    weeks without an estimate: 1
    ```

    `nc.weeklySamples` and `nc.samples` hold the samples behind the quantiles.

The two languages draw different random numbers, so their quantiles differ a little.
Their point nowcasts are the same.

Weeks end on the cut-off, so the last week is 25 to 31 August.
It has no estimate.
Under 5% of the cases with onset on 31 August are reported by then, so that day has no estimate, and nor does its week.
Set `min_reported` (`minReported` in JavaScript) lower to get one, with wide bands.

**A hub submission.**
The table is in the long format forecast hubs use for quantile forecasts.
Each row is one target date and one quantile level.
`target_end_date` is the last onset day in the week.
To submit, add the columns the hub asks for, such as the location, the reference date and the horizon.
This is how the method could take part in a hub that collects nowcasts of onsets.

**What the bands include.**
They include the random placement of each missing onset, given the estimated delay and onset curve.
They include the cases not yet reported, drawn for each onset day from a negative binomial given the number reported and $F(T - s)$.
They include uncertainty in the delay, by resampling the cases with both dates.
They do not include a delay that changes over time, onsets that are missing other than at random given the report day, or noise in the onset process itself.
If these assumptions fail, the bands are too narrow.

<figure class="doc-figure" markdown="span">
  ![Weekly onsets in the example line list from mid May to late August 2026. A dashed line counts cases with a known onset. A dotted line adds the imputed missing-onset cases and falls away in the last two weeks. The nowcast median follows the dotted line until August, then falls only from 59 to 48 a week while the dotted line falls faster. Its 50% and 95% bands widen over the last three weeks. The final week has no nowcast.](figures/nowcast.svg#only-light) ![Weekly onsets in the example line list from mid May to late August 2026. A dashed line counts cases with a known onset. A dotted line adds the imputed missing-onset cases and falls away in the last two weeks. The nowcast median follows the dotted line until August, then falls only from 59 to 48 a week while the dotted line falls faster. Its 50% and 95% bands widen over the last three weeks. The final week has no nowcast.](figures/nowcast-dark.svg#only-dark)
  <figcaption markdown="1">
    Weekly nowcast of the example line list with EM back-projection, from the Python code above.
    Weeks end on the cut-off, 31 August.
    The dashed line counts cases with a known onset.
    The dotted line adds the imputed missing-onset cases, with no correction for right truncation.
    The solid line is the nowcast median, with 50% and 95% bands.
    Bands in earlier weeks are narrow because nearly all those cases are reported.
    The last week has no nowcast, because under 5% of its last day's cases are reported.
  </figcaption>
</figure>

Read the [Assumptions](assumptions.md) page before you rely on any of this.
