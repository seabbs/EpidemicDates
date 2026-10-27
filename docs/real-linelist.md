# Using it on a real line list

This page takes an example line list in which 25% of cases have no onset date and works through it step by step.
We read the data, fill in the missing onsets with each method, and add them up by week.
Because the example is simulated, we know the true onsets, so the figures show how close each method gets.
Finally we nowcast the most recent weeks, which are still being reported, with uncertainty.
The last section uses a second example in which delays shorten over time.

The code is shown in Python, JavaScript, R and Julia.
Pick a language in any tab and the other tabs follow.
Each package's guide and API reference are on the [Packages](packages/index.md) page.

Each call returns daily onset counts after imputation under each method.
Step 6 nowcasts the recent weeks with uncertainty, using `nowcast_linelist` in Python and `nowcastLinelist` in JavaScript.
Read the [Assumptions](assumptions.md) before relying on the output.

## The example data

[Download `example-linelist.csv`](data/example-linelist.csv){ .md-button }

The file has 2,937 cases, one row per case, and three columns:

| Column | Content |
|---|---|
| `case_id` | Case identifier. |
| `onset_date` | Symptom onset date (ISO format), empty where missing. |
| `report_date` | Report date (ISO format), always present. |

```text
case_id,onset_date,report_date
case0001,2026-05-03,2026-05-05
case0002,,2026-05-07
case0003,,2026-05-12
...
case2937,2026-08-06,2026-08-06
```

745 cases (25%) have no onset date.
The data are simulated from the Python package's scenario `D`, in which fast growth turns sharply to decline on 22 July.
The data are cut at the last report date, 2026-08-06, two weeks after the turn, so recent onsets are still being reported.
The script that made the file is [`docs/data/make_example.py`](https://github.com/seabbs/EpidemicDates/blob/main/docs/data/make_example.py).

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
3044 cases, 753 missing onsets
day 0 is 2026-05-03, cut-off is day 97
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
In every language, pass a delay pmf indexed by day as `delay` to hold it fixed instead.
That also fills in the forward draw.
`delay` can also be a matrix with one pmf per onset day, for a known delay that changes over time.
Back-sampling and the mean shift always use the delays observed in the data.

!!! note "Changed behaviour"
    Before this version, `impute_linelist` in R and Julia still estimated its own EM delay when given `delay`.
    It now holds the given delay fixed, as JavaScript and the Python nowcast already did.
    Python's `impute_linelist` now takes `delay` too.

If delays change over time, estimate one delay per period with `delay_period`, or pass a known delay for each onset day.
[Delays that change over time](#delays-that-change-over-time) below shows both on a second example.

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
    2026-05-03         5          6.8     7.0     7.0
    2026-05-10        12         14.9    15.0    15.4
    2026-05-17        12         16.8    14.0    15.9
    2026-05-24        20         27.9    31.0    26.5
    2026-05-31        32         44.7    38.0    43.5
    2026-06-07        62         81.9    84.0    81.3
    2026-06-14        88        122.9   119.0   121.4
    2026-06-21       151        200.2   197.0   203.2
    2026-06-28       182        251.1   250.0   246.3
    2026-07-05       256        353.7   351.0   346.2
    2026-07-12       469        595.8   609.0   614.2
    2026-07-19       549        697.1   708.0   723.5
    2026-07-26       334        455.2   488.0   440.4
    2026-08-02       119        174.0   132.0   159.2
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
    ┌─────────┬──────────────┬──────────┬──────────────┬──────┬───────┐
    │ (index) │ week from    │ complete │ backsampling │ mean │ gen   │
    ├─────────┼──────────────┼──────────┼──────────────┼──────┼───────┤
    │ 0       │ '2026-05-03' │ 5        │ 6.8          │ 7    │ 7     │
    │ 1       │ '2026-05-10' │ 12       │ 14.9         │ 15   │ 15.4  │
    │ 2       │ '2026-05-17' │ 12       │ 16.8         │ 14   │ 15.9  │
    │ 3       │ '2026-05-24' │ 20       │ 27.9         │ 31   │ 26.5  │
    │ 4       │ '2026-05-31' │ 32       │ 44.7         │ 38   │ 43.5  │
    │ 5       │ '2026-06-07' │ 62       │ 81.9         │ 84   │ 81.3  │
    │ 6       │ '2026-06-14' │ 88       │ 122.9        │ 119  │ 121.4 │
    │ 7       │ '2026-06-21' │ 151      │ 200.2        │ 197  │ 203.2 │
    │ 8       │ '2026-06-28' │ 182      │ 251.1        │ 250  │ 246.3 │
    │ 9       │ '2026-07-05' │ 256      │ 353.7        │ 351  │ 346.2 │
    │ 10      │ '2026-07-12' │ 469      │ 595.8        │ 609  │ 614.2 │
    │ 11      │ '2026-07-19' │ 549      │ 697.1        │ 708  │ 723.5 │
    │ 12      │ '2026-07-26' │ 334      │ 455.2        │ 488  │ 440.4 │
    │ 13      │ '2026-08-02' │ 119      │ 174          │ 132  │ 159.2 │
    └─────────┴──────────────┴──────────┴──────────────┴──────┴───────┘
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
        week_from complete backsampling mean    em
    1  2026-05-03        5          6.8    7   7.0
    2  2026-05-10       12         14.9   15  15.4
    3  2026-05-17       12         16.8   14  15.9
    4  2026-05-24       20         27.9   31  26.5
    5  2026-05-31       32         44.7   38  43.5
    6  2026-06-07       62         81.9   84  81.3
    7  2026-06-14       88        122.9  119 121.4
    8  2026-06-21      151        200.2  197 203.2
    9  2026-06-28      182        251.1  250 246.3
    10 2026-07-05      256        353.7  351 346.2
    11 2026-07-12      469        595.8  609 614.2
    12 2026-07-19      549        697.1  708 723.5
    13 2026-07-26      334        455.2  488 440.4
    14 2026-08-02      119        174.0  132 159.2
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
    2026-05-03         5          6.8     7.0     7.0
    2026-05-10        12         14.9    15.0    15.4
    2026-05-17        12         16.8    14.0    15.9
    2026-05-24        20         27.9    31.0    26.5
    2026-05-31        32         44.7    38.0    43.5
    2026-06-07        62         81.9    84.0    81.3
    2026-06-14        88        122.9   119.0   121.4
    2026-06-21       151        200.2   197.0   203.2
    2026-06-28       182        251.1   250.0   246.3
    2026-07-05       256        353.7   351.0   346.2
    2026-07-12       469        595.8   609.0   614.2
    2026-07-19       549        697.1   708.0   723.5
    2026-07-26       334        455.2   488.0   440.4
    2026-08-02       119        174.0   132.0   159.2
    ```

`em` in Python, R and Julia and `gen` in JavaScript are the same method and give the same numbers here.
Each column adds the 753 missing-onset cases back, so it sums to about 3,044.
Growth turned on 22 July, and the methods part in the weeks after it.
The true onsets of the cases in the line list are 720, 448 and 157 in the last three weeks.
EM is within 8 cases of these each week.
Back-sampling is 23 too low in the week of 19 July and 17 too high in the last week, because its pooled delays come from the growth phase.
The mean shift moves every missing case by the same number of days, so it is 40 too high in the week of 26 July and 25 too low in the last week.
Over all 14 weeks the absolute errors add to 60 cases for EM, 88 for back-sampling and 119 for the mean shift.
The [interactive page](https://seabbs.github.io/EpidemicDates/interactive/) shows why.

<figure class="doc-figure" markdown="span">
  ![Weekly cases in the example line list from May to early August 2026. Grey bars show cases by report week. Lines show the true onsets, onsets with a known date, and onsets after imputation by the back-sampling, mean shift and EM methods. All curves peak in the week of 19 July and fall after the turn. In the last two weeks EM lies on the true onsets, back-sampling is above them in the final week, and the mean shift is above them and then below.](figures/linelist.svg#only-light) ![Weekly cases in the example line list from May to early August 2026. Grey bars show cases by report week. Lines show the true onsets, onsets with a known date, and onsets after imputation by the back-sampling, mean shift and EM methods. All curves peak in the week of 19 July and fall after the turn. In the last two weeks EM lies on the true onsets, back-sampling is above them in the final week, and the mean shift is above them and then below.](figures/linelist-dark.svg#only-dark)
  <figcaption markdown="1">
    Weekly counts from the example line list.
    Grey bars are cases by report week.
    The dashed line counts cases with a known onset.
    The solid black line is the true onsets of the cases in the line list.
    Coloured lines add the 753 missing-onset cases back by each method.
    The data are cut on 8 August, 17 days after growth turns, so the last weeks are incomplete under every method.
    Cases with onsets in early August are still being reported.
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
In R and Julia, `correct_truncation` gives a point nowcast without quantiles, and takes a delay matrix for a delay that changes over time.
The nowcast also takes `delay_period` (`delayPeriod`) and a delay matrix, as in [Delays that change over time](#delays-that-change-over-time).

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
    2026-07-19  2026-07-25          0.025  785.0
    2026-07-19  2026-07-25            0.1  796.0
    2026-07-19  2026-07-25           0.25  808.0
    2026-07-19  2026-07-25            0.5  820.0
    2026-07-19  2026-07-25           0.75  833.0
    2026-07-19  2026-07-25            0.9  846.0
    2026-07-19  2026-07-25          0.975  857.0
    2026-07-26  2026-08-01          0.025  573.0
    2026-07-26  2026-08-01            0.1  588.0
    2026-07-26  2026-08-01           0.25  600.0
    2026-07-26  2026-08-01            0.5  613.0
    2026-07-26  2026-08-01           0.75  628.0
    2026-07-26  2026-08-01            0.9  641.0
    2026-07-26  2026-08-01          0.975  656.0
    2026-08-02  2026-08-08          0.025  328.0
    2026-08-02  2026-08-08            0.1  351.0
    2026-08-02  2026-08-08           0.25  369.0
    2026-08-02  2026-08-08            0.5  393.0
    2026-08-02  2026-08-08           0.75  415.2
    2026-08-02  2026-08-08            0.9  440.0
    2026-08-02  2026-08-08          0.975  466.1
    weeks without an estimate: 0
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
    │ 0       │ '2026-07-19' │ '2026-07-25'    │ 0.025    │ 790   │
    │ 1       │ '2026-07-19' │ '2026-07-25'    │ 0.1      │ 799.9 │
    │ 2       │ '2026-07-19' │ '2026-07-25'    │ 0.25     │ 810   │
    │ 3       │ '2026-07-19' │ '2026-07-25'    │ 0.5      │ 820   │
    │ 4       │ '2026-07-19' │ '2026-07-25'    │ 0.75     │ 833   │
    │ 5       │ '2026-07-19' │ '2026-07-25'    │ 0.9      │ 845   │
    │ 6       │ '2026-07-19' │ '2026-07-25'    │ 0.975    │ 858   │
    │ 7       │ '2026-07-26' │ '2026-08-01'    │ 0.025    │ 574   │
    │ 8       │ '2026-07-26' │ '2026-08-01'    │ 0.1      │ 585.9 │
    │ 9       │ '2026-07-26' │ '2026-08-01'    │ 0.25     │ 598   │
    │ 10      │ '2026-07-26' │ '2026-08-01'    │ 0.5      │ 613   │
    │ 11      │ '2026-07-26' │ '2026-08-01'    │ 0.75     │ 628   │
    │ 12      │ '2026-07-26' │ '2026-08-01'    │ 0.9      │ 643   │
    │ 13      │ '2026-07-26' │ '2026-08-01'    │ 0.975    │ 655   │
    │ 14      │ '2026-08-02' │ '2026-08-08'    │ 0.025    │ 328   │
    │ 15      │ '2026-08-02' │ '2026-08-08'    │ 0.1      │ 350   │
    │ 16      │ '2026-08-02' │ '2026-08-08'    │ 0.25     │ 371   │
    │ 17      │ '2026-08-02' │ '2026-08-08'    │ 0.5      │ 391   │
    │ 18      │ '2026-08-02' │ '2026-08-08'    │ 0.75     │ 416   │
    │ 19      │ '2026-08-02' │ '2026-08-08'    │ 0.9      │ 439   │
    │ 20      │ '2026-08-02' │ '2026-08-08'    │ 0.975    │ 469   │
    └─────────┴──────────────┴─────────────────┴──────────┴───────┘
    weeks without an estimate: 0
    ```

    `nc.weeklySamples` and `nc.samples` hold the samples behind the quantiles.

The two languages draw different random numbers, so their quantiles differ a little.
Their point nowcasts are the same.

Weeks end on the cut-off, so the last week is 2 to 8 August.
Every week has an estimate here.
A day with under 5% of its cases reported gets no estimate, and nor does its week.
Set `min_reported` (`minReported` in JavaScript) lower to get one, with wide bands.

The eventual true onsets in the last three weeks are 809, 618 and 494.
The medians of the first two, 820 and 613, are close.
In the final week, where few cases are reported yet, the median of 393 is too low and the 95% band misses the truth.
The last week's interval is wide and in this draw the point estimate sits below the truth, because by chance few of its cases were reported by the cut-off.

The table has one row per week and quantile level.
`target_end_date` is the last onset day in the week.

**What the bands include.**
They include the random placement of each missing onset, given the estimated delay and onset curve.
They include the cases not yet reported, drawn for each onset day from a negative binomial given the number reported and $F(T - s)$.
They include uncertainty in the delay, by resampling the cases with both dates.
They do not include a delay that changes over time (unless you set `delay_period`, and even then not within a period), onsets that are missing other than at random given the report day, or noise in the onset process itself.
If these assumptions fail, the bands are too narrow.

<figure class="doc-figure" markdown="span">
  ![Weekly onsets in the example line list from May to early August 2026. A dashed line counts cases with a known onset. A dotted line adds the imputed missing-onset cases and falls away in the last two weeks. The nowcast median lies on the eventual true onsets up to the week of 26 July. In the final week it is 393 against a truth of 494, and its 95% band stops below the truth.](figures/nowcast.svg#only-light) ![Weekly onsets in the example line list from May to early August 2026. A dashed line counts cases with a known onset. A dotted line adds the imputed missing-onset cases and falls away in the last two weeks. The nowcast median lies on the eventual true onsets up to the week of 26 July. In the final week it is 393 against a truth of 494, and its 95% band stops below the truth.](figures/nowcast-dark.svg#only-dark)
  <figcaption markdown="1">
    Weekly nowcast of the example line list with EM back-projection, from the Python code above.
    Weeks end on the cut-off, 8 August.
    The dashed line counts cases with a known onset.
    The dotted line adds the imputed missing-onset cases, with no correction for right truncation.
    The solid line is the nowcast median, with 50% and 95% bands.
    Bands in earlier weeks are narrow because nearly all those cases are reported.
    In the final week few cases are reported yet, and this nowcast is too low.
  </figcaption>
</figure>

## Delays that change over time

Delays often shorten as a response scales up.
The second example line list, [`example-linelist-changing-delays.csv`](data/example-linelist-changing-delays.csv), has the same three columns.
Its mean delay falls from 12 days to 5 days between June and August.
The true delay of each onset day is in [`example-changing-delays-true-delay.csv`](data/example-changing-delays-true-delay.csv), because the data are simulated.

EM can use the delay in three ways:

- one static delay, estimated (the default);
- one delay per period of onset days, estimated, with `delay_period` (`delayPeriod` in JavaScript);
- a known delay for each onset day, passed as `delay` and used as given.

Periods are counted back from the cut-off, so the latest period is full length.
Each period's delay is shrunk towards the one before it, so periods with few cases borrow from their neighbour.
The [Methods](methods.md#delays-that-change-over-time) page has the maths.

This reads the second line list and the true delay, imputes with one static delay and with 14-day periods, and compares the estimated mean delay of each period with the truth.
It then gives the point nowcast of the last four weeks under each of the three delays.
In R and Julia the nowcast is `correct_truncation` with a delay for each onset day, which gives the same point nowcast.

=== "Python"

    ```python
    with open("docs/data/example-linelist-changing-delays.csv") as fh:
        rows2 = list(csv.DictReader(fh))
    onset_date2 = [to_date(r["onset_date"]) for r in rows2]
    report_date2 = [to_date(r["report_date"]) for r in rows2]
    day0_2 = min(d for d in onset_date2 + report_date2 if d is not None)
    onset2 = np.array(
        [np.nan if d is None else (d - day0_2).days for d in onset_date2]
    )
    report2 = np.array([(d - day0_2).days for d in report_date2])
    cut2 = int(report2.max())
    with open("docs/data/example-changing-delays-true-delay.csv") as fh:
        true_rows = list(csv.DictReader(fh))
    true_mean = np.array([float(r["mean_delay"]) for r in true_rows])
    true_delay = np.array([[float(r[f"p{d}"]) for d in range(61)] for r in true_rows])

    static = ed.impute_linelist(onset2, report2, cutoff=cut2)
    period = ed.impute_linelist(onset2, report2, cutoff=cut2, delay_period=14)
    days = np.arange(ed.DMAX + 1)
    print(f"static EM: {static['em_delay'] @ days:.1f}")
    print("from        true    EM")
    ends = np.r_[period["delay_period_start"][1:], cut2 + 1]
    for s, e, f in zip(period["delay_period_start"], ends, period["em_delay_by_period"]):
        date = day0_2 + dt.timedelta(days=int(s))
        print(f"{date}  {true_mean[s:e].mean():4.1f}  {f @ days:4.1f}")
    for name, opts in [
        ("static", {}),
        ("periods", {"delay_period": 14}),
        ("plug-in", {"delay": true_delay}),
    ]:
        nc = ed.nowcast_linelist(
            onset2, report2, cutoff=cut2, n_samples=1, bootstrap=False,
            min_reported=0.01, **opts
        )
        print(f"{name:<8}" + " ".join(f"{x:.1f}" for x in nc["point_weekly"][-4:]))
    ```

=== "JavaScript"

    ```js
    const { pmfMean } = require("./epidemic-dates.js");
    const readCsv = (path) => {
      const [header, ...lines] = fs.readFileSync(path, "utf8").trim().split(/\r?\n/);
      const cols = header.split(",");
      return lines.map((line) => {
        const v = line.split(",");
        return Object.fromEntries(cols.map((c, i) => [c, v[i]]));
      });
    };
    const rows2 = readCsv("docs/data/example-linelist-changing-delays.csv");
    const on2 = rows2.map((r) => (r.onset_date ? Date.parse(r.onset_date) : null));
    const rep2 = rows2.map((r) => Date.parse(r.report_date));
    const t0_2 = Math.min(...on2.filter((x) => x !== null), ...rep2);
    const onset2 = on2.map((x) => (x === null ? null : (x - t0_2) / DAY));
    const report2 = rep2.map((x) => (x - t0_2) / DAY);
    const cut2 = Math.max(...report2);
    const trueRows = readCsv("docs/data/example-changing-delays-true-delay.csv");
    const trueMean = trueRows.map((r) => Number(r.mean_delay));
    const trueDelay = trueRows.map((r) => Array.from({ length: 61 }, (_, d) => Number(r[`p${d}`])));

    const input = { onset: onset2, report: report2, cutoff: cut2 };
    const stat = imputeLinelist(input);
    const per = imputeLinelist({ ...input, delayPeriod: 14 });
    console.log(`static EM: ${pmfMean(stat.generativeDelay).toFixed(1)}`);
    console.log("from        true    EM");
    per.delayPeriodStart.forEach((s, k) => {
      const e = per.delayPeriodStart[k + 1] ?? cut2 + 1;
      const tm = trueMean.slice(s, e).reduce((a, b) => a + b, 0) / (e - s);
      const date = new Date(t0_2 + s * DAY).toISOString().slice(0, 10);
      const est = pmfMean(per.generativeDelayByPeriod[k]);
      console.log(`${date}  ${tm.toFixed(1).padStart(4)}  ${est.toFixed(1).padStart(4)}`);
    });
    for (const [name, opts] of [
      ["static", {}],
      ["periods", { delayPeriod: 14 }],
      ["plug-in", { delay: trueDelay }]
    ]) {
      const nc = nowcastLinelist({ ...input, nSamples: 1, bootstrap: false, minReported: 0.01, ...opts });
      console.log(name.padEnd(8) + nc.pointWeekly.slice(-4).map((x) => x.toFixed(1)).join(" "));
    }
    ```

=== "R"

    ```r
    rows2 <- read.csv("docs/data/example-linelist-changing-delays.csv", na.strings = "")
    onset_date2 <- as.Date(rows2$onset_date)
    report_date2 <- as.Date(rows2$report_date)
    day0_2 <- min(onset_date2, report_date2, na.rm = TRUE)
    onset2 <- as.numeric(onset_date2 - day0_2)
    report2 <- as.numeric(report_date2 - day0_2)
    cut2 <- max(report2)
    true_rows <- read.csv("docs/data/example-changing-delays-true-delay.csv")
    true_delay <- as.matrix(true_rows[, paste0("p", 0:60)])

    static <- impute_linelist(onset2, report2, cutoff = cut2)
    period <- impute_linelist(onset2, report2, cutoff = cut2, delay_period = 14)
    cat(sprintf("static EM: %.1f\n", sum(static$em_delay * 0:60)))
    cat("from        true    EM\n")
    starts <- period$delay_period_start
    ends <- c(starts[-1], cut2 + 1)
    for (k in seq_along(starts)) {
      true_k <- mean(true_rows$mean_delay[(starts[k] + 1):ends[k]])
      est_k <- sum(period$em_delay_by_period[k, ] * 0:60)
      cat(sprintf("%s  %4.1f  %4.1f\n", day0_2 + starts[k], true_k, est_k))
    }
    in_period <- findInterval(seq(0, cut2), starts)
    by_day <- list(
      static = static$em_delay,
      periods = period$em_delay_by_period[in_period, ],
      `plug-in` = true_delay
    )
    plug <- impute_linelist(onset2, report2, cutoff = cut2, delay = true_delay)
    imputed <- list(static = static, periods = period, `plug-in` = plug)
    for (name in names(by_day)) {
      nowcast <- correct_truncation(
        imputed[[name]]$imputed$em, by_day[[name]], cut2,
        min_cover = 0.01
      )
      cat(sprintf("%-8s%s\n", name, paste(sprintf("%.1f", tail(weekly(nowcast), 4)), collapse = " ")))
    }
    ```

=== "Julia"

    ```julia
    using Printf
    rows2 = split.(readlines("docs/data/example-linelist-changing-delays.csv")[2:end], ",")
    onset_date2 = [isempty(r[2]) ? missing : Date(r[2]) for r in rows2]
    report_date2 = [Date(r[3]) for r in rows2]
    day0_2 = min(minimum(skipmissing(onset_date2)), minimum(report_date2))
    onset2 = [ismissing(d) ? missing : Dates.value(d - day0_2) for d in onset_date2]
    report2 = [Dates.value(d - day0_2) for d in report_date2]
    cut2 = maximum(report2)
    true_rows = split.(readlines("docs/data/example-changing-delays-true-delay.csv")[2:end], ",")
    true_mean = [parse(Float64, r[2]) for r in true_rows]
    true_delay = permutedims(reduce(hcat, [parse.(Float64, r[3:end]) for r in true_rows]))

    static = impute_linelist(onset2, report2; cutoff = cut2)
    period = impute_linelist(onset2, report2; cutoff = cut2, delay_period = 14)
    @printf("static EM: %.1f\n", sum(static.em_delay .* (0:60)))
    println("from        true    EM")
    starts = period.delay_period_start
    ends = vcat(starts[2:end], cut2 + 1)
    for (k, s) in enumerate(starts)
        true_k = sum(true_mean[(s + 1):ends[k]]) / (ends[k] - s)
        est_k = sum(period.em_delay_by_period[k, :] .* (0:60))
        @printf("%s  %4.1f  %4.1f\n", day0_2 + Day(s), true_k, est_k)
    end
    in_period = [searchsortedlast(starts, s) for s in 0:cut2]
    plug = impute_linelist(onset2, report2; cutoff = cut2, delay = true_delay)
    for (name, fit, delay) in [("static", static, static.em_delay),
        ("periods", period, period.em_delay_by_period[in_period, :]),
        ("plug-in", plug, true_delay)]
        nowcast = correct_truncation(fit.imputed.em, delay, cut2; min_cover = 0.01)
        @printf("%-8s%s\n", name,
            join([@sprintf("%.1f", x) for x in weekly(nowcast)[(end - 3):end]], " "))
    end
    ```

```text
static EM: 8.1
from        true    EM
2026-05-03  12.0   8.5
2026-05-10  12.0  10.2
2026-05-24  12.0  12.3
2026-06-07  11.7  11.9
2026-06-21  10.2   9.6
2026-07-05   8.6   8.9
2026-07-19   6.9   7.1
2026-08-02   5.4   5.9
2026-08-16   5.0   5.4
static  487.5 458.0 421.2 444.1
periods 466.1 428.6 361.0 297.8
plug-in 460.7 420.6 356.5 320.4
```

The static delay has a mean of 8.1 days, between the early and late delays.
The 14-day periods follow the true mean down from 12 days to 5.
The first period holds only the 7 days left over at the start and has few cases, so it leans on the pooled delay.

The eventual true onsets in the last four weeks are 456, 414, 374 and 317.
The static delay is too long for recent onsets, so it divides by too small a fraction reported.
Its point nowcast is 7% to 40% too high, 18% on average over the four weeks.
With 14-day periods the error is 4% on average and at most 6%.
Passing the true delay of each onset day gives 2%.
Before the truncation correction, the absolute weekly errors of the imputed onsets add to 77 cases with periods, 85 for back-sampling, 117 for the static EM and 149 for the mean shift.

<figure class="doc-figure" markdown="span">
  ![Two panels of weekly onsets in the second example line list from May to late August 2026, where the mean delay falls from 12 to 5 days. Left, one static delay: the nowcast median rises above the eventual true onsets from late July and ends at about 440 against a truth of 317. Right, a delay for each 14-day period: the median follows the truth down to the cut-off. In both panels the dashed complete-onset line and the dotted uncorrected line fall away in the last weeks.](figures/changing-delays.svg#only-light) ![Two panels of weekly onsets in the second example line list from May to late August 2026, where the mean delay falls from 12 to 5 days. Left, one static delay: the nowcast median rises above the eventual true onsets from late July and ends at about 440 against a truth of 317. Right, a delay for each 14-day period: the median follows the truth down to the cut-off. In both panels the dashed complete-onset line and the dotted uncorrected line fall away in the last weeks.](figures/changing-delays-dark.svg#only-dark)
  <figcaption markdown="1">
    Weekly EM nowcasts of the second example line list, with 50% and 95% bands, from Python's `nowcast_linelist` with `min_reported=0.01`.
    Left: one static delay.
    Right: one delay for each 14-day period (`delay_period=14`).
    The solid black line is the eventual true onsets, the dashed line the cases with a known onset and the dotted line the onsets after imputation without the truncation correction.
    Weeks end on the cut-off, 29 August.
  </figcaption>
</figure>

**Choosing the period.**
The latest period can only show delays shorter than its length, and borrows longer ones from the period before.
Make the period longer than most delays, but short enough to follow the change.
Here 28-day periods do about as well as 14-day ones.
`delay_shrinkage` (`delayShrinkage`) sets how strongly each period leans on the one before; the default of 10 expected cases rarely needs changing.

**A known delay.**
If you know the delay of each onset day, for example from a study of the reporting process, pass it as `delay`: a matrix (in JavaScript, an array of arrays) with one pmf per onset day `0..cutoff`.
It is used as given by EM, the forward draw and the truncation correction, and nothing is re-estimated.
Back-sampling and the mean shift still use the delays in the data.

Read the [Assumptions](assumptions.md) page before you rely on any of this.
