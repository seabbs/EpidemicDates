# Methods

This page sets out the maths behind each method.
The [interactive page](https://seabbs.github.io/epidemic-imputation-demonstration/) shows the same maths under "How the methods work".
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

## Hub: pooled empirical delays

The hub method (`hub`) subtracts a delay drawn from the pooled empirical delays of complete cases reported by the cut-off:

$$
P(s \mid t) = g(t - s),
\qquad
g(d) \propto \sum_{s \le T - d} C(s) \, f_s(d) \, \big(1 - \pi(s + d, d)\big).
$$

The same $g$ is used for every report day $t$.

**Steady growth.**
If $C(s) \propto e^{rs}$ with $r > 0$, and the delay and $\pi$ are constant, the correct posterior is proportional to $f(d) e^{-rd}$.
This is the backward delay.
The sum in $g$ is also proportional to $e^{-rd}$ when $T$ is well after the start of the epidemic, so $g(d) \propto f(d) e^{-rd}$ and the hub draw matches the truth.
When growth changes, the two differ.
The hub method then shifts missing onsets into the wrong weeks.

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
Without a given delay, `impute_linelist` in Python and `imputeLinelist` in JavaScript also estimate a single static delay $f$:

$$
f(d) \propto \frac{n_c(d) + n_a(d)}{\sum_{s \le T - d} \lambda(s)},
$$

where $n_c(d)$ and $n_a(d)$ count complete and allocated cases with delay $d$.
The denominator allows for right truncation, since a delay of $d$ can only be seen for onsets up to $T - d$.
The code runs a fixed number of iterations rather than testing convergence.

**On the interactive page** the generative method is given the true delay and noise-free expected counts.
That makes it a best case.

### What a full model would add

EM back-projection here is a demonstration.
A full generative model, such as [epinowcast](https://package.epinowcast.org/articles/model.html#accounting-for-reported-cases-with-a-missing-reference-date), would:

- estimate a delay that changes over time, jointly with the onset curve;
- smooth the onset curve, for example with a renewal process, so small counts do not give noisy output;
- model ascertainment and missingness as part of the observation process;
- report uncertainty rather than one expected allocation.

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
The hub draw and EM back-projection keep the right shape near the cut-off, so the correction works for them.

The generative method has the same correction built in.
Its onset curve $\lambda(s) = \big(n_c(s) + n_a(s)\big) / F_s\big(\min(T - s, D)\big)$ divides by the fraction reported in the same way.

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

The hub method draws observed whole-day delays, which need no correction.
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
