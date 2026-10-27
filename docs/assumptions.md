# Assumptions

These are the assumptions behind the simulations on the [interactive page](https://seabbs.github.io/epidemic-imputation-demonstration/) and in the Python package.
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
    The generative method estimates one delay for the whole data vintage.
    If delays shorten as the response scales up, supply a delay or use a model with a time-varying delay.

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

For production use, a model that treats missing onsets as part of the observation process is the better option.
An example is [epinowcast](https://package.epinowcast.org/articles/model.html#accounting-for-reported-cases-with-a-missing-reference-date).
