# Methods

This page sets out the maths behind each method.
The [interactive page](https://seabbs.github.io/EpidemicDates/interactive/) shows the same maths under "How the methods work".
What each result relies on is listed under [Assumptions](assumptions.md).

## Notation

A case with a missing onset is reported on day $t$.
Its unknown onset day is $s$ and its delay is $d = t - s$.

| Symbol | Meaning |
|---|---|
| $T$ | Data cut-off: the last report day in the data. |
| $D$ | Maximum delay (`D` in JavaScript, `DMAX` in Python). |
| $C(s)$ | Expected confirmed onsets on day $s$. |
| $f_s(d)$ | Forward onset-to-report delay pmf for onset day $s$, on $0, \dots, D$. Written $f(d)$ when it does not change over time. |
| $F_s$ | The cumulative distribution function (CDF) of $f_s$. |
| $\pi(t, d)$ | Probability that the onset of a case reported on day $t$ with delay $d$ is missing. |

Cases with a known onset keep it under every method.
Only the cases with a missing onset are imputed.

## The correct posterior

Given the report day, the onset day of a missing case has posterior

$$
P(s \mid t) \propto C(s) \, f_s(t - s) \, \pi(t, t - s).
$$

It depends on how many onsets there were on each earlier day.
A method is unbiased when it reproduces this distribution.

## Assumptions of each method

Every method keeps known onsets and only imputes the missing ones.
Each relies on more than it seems.
The rest of this page gives the maths and shows where each assumption breaks.

**Back-sampling** (`backsampling`) assumes:

- onsets are missing at random given the report day;
- delays are the same over time and place;
- growth is steady over the span of the delay, so the pooled empirical delay matches the backward delay;
- the cut-off is well after the start of the epidemic.

It breaks when growth turns, when delays shorten or lengthen, and when cases with long delays are more likely to miss an onset.

**Mean shift** (`mean`) assumes all of the above and that every case has the same delay.
It breaks whenever delays vary, which they always do.
Near the cut-off it leaves the last few days with no imputed cases.

**Forward draw** (`fwd`, `forward`) assumes:

- onsets are missing at random given the report day;
- the forward delay is known, or estimated with a correction for right truncation;
- onsets are flat over the span of the delay, since it ignores how many onsets there were on each day.

It breaks when the epidemic grows or declines.

**Generative method** (`gen`, `em`) assumes:

- onsets are missing at random given the report day;
- the delay is known, or is estimated jointly with the onset curve, allowing for right truncation, either as one static delay or as one delay per period of onset days;
- counts are large enough, as there is no smoothing.

It breaks when delays change over time but are estimated as static, when they change within a period, when missingness depends on the delay, and gives noisy output with small counts.

## Which delay each method uses

- **Back-sampling** and the **mean shift** use the delays observed in the data: the pooled empirical delays of complete cases. They never take a supplied delay.
- The **forward draw** uses the supplied delay as given (a plug-in). Without one it is not run on a line list, except in the nowcast, where it uses the EM estimate.
- **EM back-projection** has three modes:
    1. a supplied delay, held fixed as a plug-in, either one pmf or one pmf per onset day;
    2. no delay and no `delay_period` (the default): one static delay, estimated;
    3. `delay_period` (`delayPeriod` in JavaScript): one delay per period, estimated.
- The **truncation correction** in the nowcast uses the same delay as EM: the supplied one, or the estimate.

A supplied delay is never re-estimated, and it cannot be combined with `delay_period`.
This is the same in all four languages.
Before this version, `impute_linelist` in R and Julia still estimated its own EM delay when given `delay`.

## Back-sampling: pooled empirical delays

The back-sampling method (`backsampling`) subtracts a delay drawn from the pooled empirical delays of complete cases reported by the cut-off:

$$
P(s \mid t) = g(t - s),
\qquad
g(d) \propto \sum_{s \le T - d} C(s) \, f_s(d) \, \big(1 - \pi(s + d, d)\big).
$$

The same $g$ is used for every report day $t$.

**Steady growth.**
If $C(s) \propto e^{rs}$ with $r > 0$, and the delay and $\pi$ are constant, the correct posterior is proportional to $f(d) e^{-rd}$.
This is the backward delay.
The sum in $g$ is also proportional to $e^{-rd}$ when $T$ is well after the start of the epidemic, so $g(d) \propto f(d) e^{-rd}$ and the back-sampling draw matches the truth.
When growth changes, the two differ.
The back-sampling method then shifts missing onsets into the wrong weeks.

<figure class="doc-figure" markdown="span">
  ![Two panels of delay probability against delay in days. Left, steady growth: the pooled empirical delay lies on top of the correct backward delay, and both are shorter than the forward delay. Right, ten days after growth turns: the pooled empirical delay is shorter than the correct backward delay, which has extra weight around 10 days.](figures/delays.svg#only-light) ![Two panels of delay probability against delay in days. Left, steady growth: the pooled empirical delay lies on top of the correct backward delay, and both are shorter than the forward delay. Right, ten days after growth turns: the pooled empirical delay is shorter than the correct backward delay, which has extra weight around 10 days.](figures/delays-dark.svg#only-dark)
  <figcaption markdown="1">
    Delay distributions for a case reported on the cut-off day, from expected counts.
    Left: scenario `A`, steady growth at 0.05 per day, cut at day 100.
    Right: scenario `D`, growth at 0.06 per day that turns to −0.04 at day 80, cut at day 90.
    The forward delay has mean 8 days and SD 8 days.
    In steady growth the pooled empirical delay matches the correct backward delay.
    After the turn it still reflects the growth phase.
    The correct delay is longer, because more onsets happened around the turn, 10 days before the report.
  </figcaption>
</figure>

## Mean shift

The mean shift (`mean`) moves every missing case back by the rounded mean of the pooled empirical delay:

$$
s = t - \operatorname{round}\Big(\sum_d d \, g(d)\Big).
$$

Every missing case moves back by the same amount, so the shape of the delay is lost.

## Forward draw

The forward draw (`fwd` in JavaScript, `forward` in Python) uses the forward delay:

$$
P(s \mid t) = \bar f(t - s).
$$

In JavaScript $\bar f$ is the onset-weighted forward delay of the scenario.
In Python it is the true forward delay.
This ignores how many onsets there were on each day.
It stands in for a delay estimated with a correction for right truncation.
In simulation it is an oracle, because the true delay is known.

## Generative method: EM back-projection

The generative method (`gen` in JavaScript, `em` in Python) is EM back-projection.
It is a partial deconvolution of the missing-onset cases.

Complete cases keep their onsets.
Each missing case reported on day $t$ is shared over onset days in proportion to the current onset curve $\lambda$ and the delay:

$$
P(s \mid t) \propto \lambda(s) \, f_s(t - s).
$$

The onset curve is then re-estimated from the complete and allocated cases, and divided by the fraction reported by the cut-off:

$$
\lambda(s) = \frac{n_c(s) + n_a(s)}{F_s\big(\min(T - s, D)\big)}.
$$

Here $n_c(s)$ counts complete cases and $n_a(s)$ allocated cases with onset day $s$.
The division by $F_s$ allows for right truncation.
The two steps repeat until $\lambda$ settles.
This is the EM algorithm for the likelihood with the missing onsets as latent variables.
It is the same idea as Becker's back-projection of HIV infections and Richardson–Lucy deconvolution.

If onsets are missing at random given the report day, so that $\pi$ depends only on $t$, the fixed point matches the correct posterior.

**Estimating the delay.**
On a real line list the delay is unknown.
Without a given delay, `impute_linelist` (`imputeLinelist` in JavaScript) also estimates a single static delay $f$:

$$
f(d) \propto \frac{n_c(d) + n_a(d)}{\sum_{s \le T - d} \lambda(s)},
$$

where $n_c(d)$ and $n_a(d)$ count complete and allocated cases with delay $d$.
The denominator allows for right truncation, since a delay of $d$ can only be seen for onsets up to $T - d$.
The code runs a fixed number of iterations rather than testing convergence.

### Delays that change over time

Delays often change over an outbreak, for example falling as a response scales up.
One static delay is then too long late on and too short early on.
Back-projection places recent missing onsets too early, and the nowcast over-corrects the latest weeks.

**Periods.**
With `delay_period` $P$, onset days are split into $K = \lceil (T + 1) / P \rceil$ periods counted back from the cut-off.
Onset day $s$ is in period

$$
k(s) = K - 1 - \left\lfloor \frac{T - s}{P} \right\rfloor .
$$

The latest period is always $P$ days long.
The first period holds the days left over and can be shorter.
Each period $k$ has its own delay $f_k$, so $f_s = f_{k(s)}$ in the notation above.

**E-step.** A missing case reported on day $t$ is shared over onset days by

$$
P(s \mid t) \propto \lambda(s) \, f_{k(s)}(t - s).
$$

**M-step.** The onset curve divides by the fraction reported under each day's own delay,

$$
\lambda(s) = \frac{n_c(s) + n_a(s)}{F_{k(s)}\big(\min(T - s, D)\big)}.
$$

The pooled delay $f$ is updated as for the static delay.
Then, in order $k = 0, 1, \dots, K - 1$,

$$
f_k(d) \propto \frac{N_k(d) + \kappa \, h_k(d)}{E_k(d) + \kappa},
\qquad
h_0 = f,
\quad
h_k = f_{k-1},
$$

where

$$
N_k(d) = \sum_{s \in k} \big(n_c(s, d) + n_a(s, d)\big),
\qquad
E_k(d) = \sum_{s \in k,\; s \le T - d} \lambda(s).
$$

$N_k(d)$ counts complete and allocated cases with onset in period $k$ and delay $d$.
$E_k(d)$ is the expected number of period-$k$ onsets that could have been seen with delay $d$ by the cut-off, so it allows for right truncation within the period.
Without $\kappa$ this is the static update restricted to one period.
Each $f_k$ is floored at $10^{-12}$ and renormalised.
The E-step and M-step alternate as before, with three M-step passes per iteration.

**Shrinkage.**
$\kappa$ is `delay_shrinkage` (`delayShrinkage` in JavaScript), 10 expected cases by default.
The weight on a period's own data at delay $d$ is $E_k(d) / \big(E_k(d) + \kappa\big)$, which grows with the number of cases the period could show at that delay.
The update is the posterior mode of $f_k(d)$ under a $\operatorname{Gamma}\big(\kappa h_k(d) + 1, \kappa\big)$ prior, whose mode is $h_k(d)$.
So the periods follow a random walk: each is pulled towards the one before it.

This matters most in two places.
A period with no cases has $E_k = N_k = 0$ and takes the previous period's delay.
The latest period can only show delays up to $P - 1$ days, so for longer delays $E_k(d) = 0$ and it takes the previous period's values.
The same holds, to a lesser degree, for every period near the cut-off.

We shrink towards the previous period rather than the pooled delay because it is closer in time.
In scenario `F` (growth at 0.03 per day, mean delay falling from 12 to 4 days), cut at day 149, over 20 simulated line lists, the mean absolute error of the weekly point nowcast over the last four weeks was:

| Delay | Mean absolute error |
|---|---|
| Static (default) | 12.2% |
| 14-day periods, shrunk to the pooled delay | 8.3% |
| 14-day periods, shrunk to the previous period | 4.8% |
| 28-day periods, shrunk to the previous period | 5.7% |

With the true delay of each onset day supplied, the error was 2.6%.
The remaining gap comes from the latest period's long delays, which are borrowed from a period whose delay was longer.
At a cut-off of day 100 the errors were 11.0% static and 7.4% with 14-day periods.
The error in the imputed onsets, before the truncation correction, fell from 3.0% to 2.7% of all cases at day 149.
At day 100 it was 5.6% either way, as most of that error is noise.
$\kappa$ between 1 and 10 gave similar results; $\kappa = 100$ was worse.

<figure class="doc-figure" markdown="span">
  ![Two panels of weekly onsets from day 80 to day 149 in a simulation where the mean delay falls from 12 to 4 days. Left, one static delay: the nowcast median rises above the eventual true onsets in the last three weeks, reaching about 1270 against a truth of 980 in the final week. Right, a delay for each 14-day period: the median stays closer to the truth, about 1130 in the final week. In both panels the dashed complete-onset line falls away near the cut-off.](figures/nowcast-periods.svg#only-light) ![Two panels of weekly onsets from day 80 to day 149 in a simulation where the mean delay falls from 12 to 4 days. Left, one static delay: the nowcast median rises above the eventual true onsets in the last three weeks, reaching about 1270 against a truth of 980 in the final week. Right, a delay for each 14-day period: the median stays closer to the truth, about 1130 in the final week. In both panels the dashed complete-onset line falls away near the cut-off.](figures/nowcast-periods-dark.svg#only-dark)
  <figcaption markdown="1">
    Weekly EM nowcasts of one scenario `F` line list, cut at day 149, with 50% and 95% bands.
    Left: one static delay.
    Right: one delay for each 14-day period.
    The static delay is too long for recent onsets, so it divides by too small a fraction reported and overshoots.
    Per-period delays halve the overshoot in the final week.
    Some remains, because the latest period borrows its longer delays from the period before, and the bands still miss the truth there.
  </figcaption>
</figure>

**Choosing the period.**
Shorter periods follow the delay more closely but have fewer cases each, so they lean more on the shrinkage.
The latest period cannot show delays longer than $P - 1$ days, so make $P$ longer than most delays.
With `delay_period` at least $T + 1$ there is one period and the result is the same as the default.

**A supplied delay that changes over time.**
If the delay of each onset day is known, pass `delay` as a matrix with one row per onset day $0, \dots, T$.
EM then holds $f_s$ fixed and only estimates $\lambda$.
The forward draw uses $P(s \mid t) \propto f_s(t - s)$ over onset days $0, \dots, t$.
The nowcast divides by $F_s(T - s)$.
A delay known per period can be passed this way by repeating each period's pmf for its days.

**On the interactive page** the generative method is given the true delay and noise-free expected counts.
That makes it a best case.

### What a full model would add

EM back-projection here is a demonstration.
A full generative model, such as [epinowcast](https://package.epinowcast.org/articles/model.html#accounting-for-reported-cases-with-a-missing-reference-date), would:

- estimate a delay that changes smoothly over time, jointly with the onset curve, rather than one per fixed period;
- smooth the onset curve, for example with a renewal process, so small counts do not give noisy output;
- model ascertainment and missingness as part of the observation process;
- report uncertainty rather than one expected allocation.

## Where each method places one case

This compares the four methods for one report day just after a turn in growth.
The weekly effect is on the [home page](index.md#bias-at-a-turn-in-growth).

<figure class="doc-figure" markdown="span">
  ![Probability over onset days 55 to 90 for one missing case reported on day 90, after growth turned at day 80. The generative method lies on the truth, with mean day 82.7. The back-sampling draw is shifted later, with mean day 84.2. The forward draw is shifted earlier, with mean day 82.0. The mean shift puts all its weight on day 84.](figures/placement.svg#only-light) ![Probability over onset days 55 to 90 for one missing case reported on day 90, after growth turned at day 80. The generative method lies on the truth, with mean day 82.7. The back-sampling draw is shifted later, with mean day 84.2. The forward draw is shifted earlier, with mean day 82.0. The mean shift puts all its weight on day 84.](figures/placement-dark.svg#only-dark)
  <figcaption markdown="1">
    Where each method places the onset of one missing case reported on day 90.
    Scenario `D`, with the data cut at day 90, 10 days after growth turns.
    Truth is the correct posterior $P(s \mid t)$.
    The generative method is given the true delay, so it matches the truth.
    The back-sampling draw places onsets too late and the forward draw too early.
    The mean shift puts every case on one day.
    The legend gives each method's mean onset day.
  </figcaption>
</figure>

## Comparison modes

Every method gives imputed onsets on day $s$,

$$
\hat n(s) = n_c(s) + n_i(s),
$$

where $n_c(s)$ counts complete cases and $n_i(s)$ the missing-onset cases each method places on day $s$.
What $\hat n$ is compared with depends on the mode.
The JavaScript functions are `comparisonSeries`, `correctTruncation` and `completeWeeks`.

### Retrospective: missing onsets only

The data are cut after every case has been reported.
The only problem is the missing onsets.
The truth is all onsets,

$$
C(s) = \sum_d C(s) \, f_s(d),
$$

and each method is compared with it as is.
Bias comes only from where each method places the missing cases.

### Real time with a truncation correction

This is the default on the interactive page.
The data are cut at $T$, so $\hat n(s)$ counts only cases reported by $T$.
Each method is corrected by the expected fraction of day $s$ onsets reported by $T$:

$$
F_s(T - s) = \sum_{d \le T - s} f_s(d),
\qquad
\hat C(s) = \frac{\hat n(s)}{F_s(T - s)}.
$$

The correction uses the true delay, as if the hub had done this correctly.
Days with $F_s(T - s) < 0.05$ are set to 0.
The truth is all onsets that will eventually be reported, $C(s)$.
A week is complete when its case-weighted fraction reported, $\sum_{s \in w} C(s) F_s(T - s) / \sum_{s \in w} C(s)$, is at least 30%.
Weeks below that are shaded on the page and left out of the growth and peak metrics.

### Real time without a correction

With the nowcast off, $\hat n(s)$ is compared with the onsets of cases reported by $T$.
Recent weeks are then incomplete for the truth and every method alike.
Weeks under 80% reported are shaded and left out of the growth and peak metrics.

### Why the correction cannot rescue a misplacement

The correction rescales each onset day by its own factor $1 / F_s(T - s)$.
It assumes the imputed curve has the right shape and is only incomplete.
If a method puts missing cases on the wrong day, the correction scales up the error.

The mean shift moves every missing case back by $m$ days.
The last $m$ days before $T$ then get no missing cases at all.
The correction scales these days up the most, so the error grows.
At the baseline scenario the mean shift is about 12% too low in the final week.
The back-sampling draw and EM back-projection keep the right shape near the cut-off, so the correction works for them.

<figure class="doc-figure" markdown="span">
  ![Two panels of expected daily onsets for days 55 to 90 with a cut-off at day 90, in steady growth. Left, backsampling: the corrected curve lies on the eventual truth, while the uncorrected curve falls towards the cut-off. Right, mean shift: the corrected curve overshoots the truth up to day 84, then drops well below it for the last 6 days.](figures/truncation.svg#only-light) ![Two panels of expected daily onsets for days 55 to 90 with a cut-off at day 90, in steady growth. Left, backsampling: the corrected curve lies on the eventual truth, while the uncorrected curve falls towards the cut-off. Right, mean shift: the corrected curve overshoots the truth up to day 84, then drops well below it for the last 6 days.](figures/truncation-dark.svg#only-dark)
  <figcaption markdown="1">
    Expected daily onsets near a cut-off at day 90 in scenario `A`, steady growth at 0.05 per day.
    Dashed: onsets after imputation of cases reported by the cut-off.
    Solid: the same divided by $F(T - s)$.
    Black: the eventual truth.
    The back-sampling draw has the right shape in steady growth, so the correction recovers the truth.
    The mean shift moves every missing case back 6 days and leaves none in the last 6 days.
    The correction then overshoots before day 85 and undershoots after it.
  </figcaption>
</figure>

The generative method has the same correction built in.
Its onset curve $\lambda(s) = \big(n_c(s) + n_a(s)\big) / F_s\big(\min(T - s, D)\big)$ divides by the fraction reported in the same way.

## Nowcast with uncertainty

`nowcast_linelist` in Python and `nowcastLinelist` in JavaScript turn a line list into a nowcast with quantiles.
Write $f_s$ for the delay of onset day $s$: the same for every day with one static delay, $f_{k(s)}$ with `delay_period`, or row $s$ of a supplied delay matrix.
They impute, correct for right truncation and then sample.

**Point nowcast.**
Impute the missing onsets with the chosen method, giving $\hat n(s) = n_c(s) + n_i(s)$.
The default is EM back-projection.
The delay is the given one, or the delay estimated by EM allowing for right truncation, static or one per period.
Then

$$
\hat C(s) = \frac{\hat n(s)}{F_s\big(\min(T - s, D)\big)}.
$$

Days with $F_s(T - s)$ below a minimum (5% by default) get no estimate.
A week with any such day has no estimate either.
Weeks end on the cut-off.

**Samples.**
Each sample $b$ does three things.

1. **Delay.** Resample the complete cases with replacement.
   Recompute the pooled empirical delay $g_b$.
   Refit EM to the resampled complete cases and all missing cases, starting from the point fit, to get $f_{b,s}$ and $\lambda_b$.
   With `delay_period` this refits one delay per period.
   With a given delay, $f_{b,s} = f_s$ and only $\lambda_b$ is refitted.
   This step is optional.
2. **Missing onsets.** For the $m(t)$ missing cases reported on day $t$, draw onsets from

    $$
    (m_{t,s})_s \sim \operatorname{Multinomial}\big(m(t), P_b(s \mid t)\big),
    $$

    with $P_b(s \mid t) = g_b(t - s)$ for back-sampling, $f_b(t - s)$ for the forward draw and $\propto \lambda_b(s) f_{b,s}(t - s)$ for EM.
    When the delay changes over time the forward draw uses $\propto f_{b,s}(t - s)$ over onset days $0, \dots, t$.
    The mean shift moves every case back by the rounded mean of $g_b$, with no draw.
    Then $n_b(s) = n_c(s) + \sum_t m_{t,s}$.
3. **Not yet reported.** Draw the cases with onset $s$ still to be reported,

    $$
    U_b(s) \sim \operatorname{NegBin}\big(n_b(s), F_{b,s}(T - s)\big),
    \qquad
    C_b(s) = n_b(s) + U_b(s).
    $$

    $U_b(s)$ counts failures before $n_b(s)$ successes with success probability $F_{b,s}(T - s)$, the CDF of onset day $s$'s delay.
    Its mean is $n_b(s) \big(1 - F_{b,s}(T - s)\big) / F_{b,s}(T - s)$, so $E[C_b(s)] = n_b(s) / F_{b,s}(T - s)$, the same correction as the point nowcast.
    It is the posterior for the eventual total when each case is reported by $T$ with probability $F_{b,s}(T - s)$ and the prior on the total is proportional to $1 / C(s)$.

Weekly samples sum the daily samples.
Quantiles at 0.025, 0.1, 0.25, 0.5, 0.75, 0.9 and 0.975 come from the samples by linear interpolation.
They are returned with one row per target date and quantile.

**Captured:** the random placement of missing onsets; the binomial reporting of recent onsets; and, with resampling, sampling uncertainty in the delay and in the EM onset curve.

**Not captured:** a delay that changes over time but is estimated as static, or that changes within a period; missingness that depends on the delay or on anything other than the report day; dependence between cases; uncertainty in a given delay; and noise in the onset process itself, since the target is the onsets that will eventually be reported.
A day with no reported onsets gets 0 in every sample.
With small counts the median is below the mean, so medians can sit low.

<figure class="doc-figure" markdown="span">
  ![Weekly onsets in the example line list from May to early August 2026. A dashed line counts cases with a known onset. A dotted line adds the imputed missing-onset cases and falls away in the last two weeks. The nowcast median lies on the eventual true onsets up to the week of 26 July. In the final week it is 393 against a truth of 494, and its 95% band stops below the truth.](figures/nowcast.svg#only-light) ![Weekly onsets in the example line list from May to early August 2026. A dashed line counts cases with a known onset. A dotted line adds the imputed missing-onset cases and falls away in the last two weeks. The nowcast median lies on the eventual true onsets up to the week of 26 July. In the final week it is 393 against a truth of 494, and its 95% band stops below the truth.](figures/nowcast-dark.svg#only-dark)
  <figcaption markdown="1">
    Weekly nowcast of the [example line list](real-linelist.md) with EM back-projection and one static delay.
    The solid black line is the eventual true onsets, known because the data are simulated.
    The dashed line counts cases with a known onset.
    The dotted line adds the imputed missing-onset cases, with no correction for right truncation.
    The coloured line is the nowcast median, with 50% and 95% bands.
    In the final week few cases are reported yet, and this nowcast is too low.
  </figcaption>
</figure>

## Delay distribution

Delays are double interval censored daily Gamma pmfs, in the form used by [primarycensored](https://primarycensored.epinowcast.org).
The primary event is uniform within its day and the secondary event is censored to its day.
Then

$$
P(d) = G(d + 1) - 2 G(d) + G(d - 1),
$$

$$
G(x) = \int_0^x F(u; k, \theta) \, du = x F(x; k, \theta) - k \theta F(x; k + 1, \theta),
$$

where $F(x; k, \theta)$ is the Gamma CDF with shape $k$ and scale $\theta$, and $G(x) = 0$ for $x \le 0$.
For mean $\mu$ and coefficient of variation $c$, $k = 1 / c^2$ and $\theta = \mu c^2$.
The pmf is truncated at $D$ and renormalised, so its mean is close to $\mu$.

The back-sampling method draws observed whole-day delays, which need no correction.
A parametric delay fitted to data would need double interval censoring and a right-truncation adjustment, for example with [primarycensored](https://primarycensored.epinowcast.org) or [epidist](https://epidist.epinowcast.org).

## Growth rate and Rt

Weekly counts are turned into a growth rate per day for week $w$:

$$
r_w = \frac{\log x_{w+1} - \log x_{w-1}}{14},
$$

with a one-sided difference in the first and last weeks.
Rt follows from a Gamma generation interval with shape $k_g$ and scale $\theta_g$ (Wallinga and Lipsitch):

$$
R = (1 + r \theta_g)^{k_g},
\qquad
k_g = \left(\frac{m_g}{s_g}\right)^2,
\qquad
\theta_g = \frac{s_g^2}{m_g}.
$$

The generation interval has mean $m_g = 15$ days and standard deviation $s_g = 9$ days, an Ebola-like value.


## Related tools

[backcalc](https://github.com/epiforecasts/backcalc) and [EpiNow2](https://epiforecasts.io/EpiNow2/) work on a related problem.
There no reference dates (such as onsets) are reported at all.
Back-sampling from report dates must then reconstruct the whole onset curve, not only the missing part.
The same bias at turns in growth applies, and to every case rather than a fraction.
EpiNow2 also has a generative model that convolves an infection curve with the delays instead.

[epinowcast](https://package.epinowcast.org) nowcasts jointly with missing reference dates.
It treats cases without a reference date as part of the observation process, rather than imputing them first.
