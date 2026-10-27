# Assumptions

These are the assumptions behind the simulations on the [interactive page](https://seabbs.github.io/EpidemicDates/interactive/) and in the Python package.
The maths is on the [Methods](methods.md) page.

## In the simulations

**The back-sampling method.**
We assume each missing onset is drawn from the raw empirical delays of cases with both dates.
The delays are pooled over the whole data vintage, with no correction for right truncation and no stratification by time or place.
This follows the description in [issue #61](https://github.com/InsightNet-US/BDBV-Modeling-Hub/issues/61) but is not confirmed.
If the delays were corrected for truncation, the method would behave like the forward draw.

**The target.**
By default the data are cut at the last day.
Every method is then corrected for right truncation using the true delay, as if the hub had done this correctly.
Truth is all onsets that will eventually be reported.
With the nowcast off, truth is the onsets of cases reported by the cut-off.
Recent weeks are then incomplete for the truth and every method alike.
In real data the delay used for the correction would itself have to be estimated.

**Missingness.**
Onsets are missing at random given the report day.
The exception is when missingness is set to depend on the delay ("Extra missingness with long delays" on the page, scenario `G` in Python).

**Delays.**
Delays are double interval censored daily Gamma.
They are the same for every case unless the mean is set to change over time.
Real delays have heavier tails.

**The generative method.**
On the page it is given the true delay and noise-free expected counts, so it is a best case.
The Python cross-check estimates the delay jointly with the onset curve and stays close to unbiased.

**Which delay each method uses.**
Back-sampling and the mean shift use the delays observed in the data.
The forward draw uses the supplied delay.
EM back-projection holds a supplied delay fixed, and otherwise estimates one static delay or, with `delay_period`, one delay per period.
The [Methods](methods.md#which-delay-each-method-uses) page sets this out.

**Outputs.**
Results are expected values unless stochastic replicates are used.
Counts are summed to onset weeks.
Rt uses a Gamma generation interval with mean 15 days and SD 9 days.

## On a real line list

These apply when you use `impute_linelist` or `imputeLinelist` on your own data.
The [worked example](real-linelist.md) shows how.

!!! note "Missingness"
    All methods assume onsets are missing at random given the report day.
    Check whether cases without onsets differ, for example by delay, place or outcome.
    If they do, every method here is biased.

!!! note "Delays changing over time"
    By default the generative method estimates one delay for the whole data vintage.
    If delays shorten as the response scales up, set `delay_period` (`delayPeriod` in JavaScript) to estimate one delay per period of onset days.
    Or supply a known delay for each onset day as a matrix.
    What is still not handled:

    - change within a period, as each period has one delay;
    - the longest delays in the latest periods, which are borrowed from the period before, so a falling delay is still overestimated there;
    - periods with few cases, which lean on their neighbour through the shrinkage;
    - missingness that depends on the delay, which biases the delay of each period as it does the static one.

    A model with a smoothly time-varying delay, such as epinowcast, handles the first better.

!!! note "A supplied delay"
    A delay passed as `delay` is used as given by EM, the forward draw and the truncation correction.
    It is never re-estimated.
    Before this version, R and Julia's `impute_linelist` estimated their own EM delay even when given one.

!!! note "Censoring and truncation"
    If you fit your own parametric delay to pass as `delay`, fit it with double interval censoring and right truncation.
    For example, use [primarycensored](https://primarycensored.epinowcast.org) or [epidist](https://epidist.epinowcast.org).

!!! note "Noise"
    EM back-projection has no smoothing.
    With small counts its daily output is noisy.
    Aggregate to weeks, or use a full model.

!!! note "Recent days"
    Onsets in the last weeks before the cut-off are incomplete under every method.
    Cases with those onsets are still being reported.
    `nowcast_linelist` and `nowcastLinelist` correct for this.

For production use, a model that treats missing onsets as part of the observation process is the better option.
An example is [epinowcast](https://package.epinowcast.org/articles/model.html#accounting-for-reported-cases-with-a-missing-reference-date).

## In the nowcast

These apply when you use `nowcast_linelist` in Python or `nowcastLinelist` in JavaScript.
The maths is under [Nowcast with uncertainty](methods.md#nowcast-with-uncertainty).

!!! note "The delay for the correction"
    The truncation correction divides by $F_s(T - s)$, the CDF of onset day $s$'s delay.
    The delay is given, or estimated by EM with right truncation: one static delay by default, or one per period with `delay_period`.
    If delays change over the data vintage and one static delay is used, recent days are over- or under-corrected.
    With periods, change within the latest period, and its longest delays, are still taken from earlier data.

!!! note "Reporting"
    Each case with onset on day $s$ is reported by the cut-off with the same probability $F(T - s)$, independently of other cases.
    The cases not yet reported are then negative binomial given those reported.
    Batches of late reports or backlogs break this, and the bands are then too narrow.

!!! note "Days with little reported"
    Days with under 5% of cases reported get no estimate, and nor do weeks that contain them.
    Change this with `min_reported` (`minReported` in JavaScript).
    A day with no reported onsets gets 0 in every sample.
    With small counts the median sits below the mean.

!!! note "What the bands leave out"
    The bands include the draw of missing onsets, the cases not yet reported and, by resampling cases with both dates, uncertainty in the delay.
    They leave out changes in the delay over time (within a period, with `delay_period`), missingness that is not at random given the report day, uncertainty in a given delay, and noise in the onset process itself.
    Treat them as a lower bound on the uncertainty.
