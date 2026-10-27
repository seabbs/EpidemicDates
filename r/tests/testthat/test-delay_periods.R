# Delays that change over time: per-period EM delays and delay matrices.

# A Poisson line list with steady growth and a mean delay falling linearly
# from 12 to 5 days, cut at `cutoff`. 40% of onsets are missing.
simulate_falling <- function(seed, cutoff = 120) {
  set.seed(seed)
  days <- seq(0, cutoff)
  means <- 12 - 7 * days / cutoff
  delay <- t(vapply(means, function(m) delay_pmf(m, 0.8 * m), numeric(61)))
  expected <- 30 * exp(0.03 * days) * delay
  n <- matrix(rpois(length(expected), expected), nrow(expected))
  onset <- rep(rep(days, ncol(n)), as.vector(n))
  report <- onset + rep(rep(seq(0, 60), each = nrow(n)), as.vector(n))
  eventual <- rowSums(n)
  keep <- report <= cutoff
  onset <- onset[keep]
  report <- report[keep]
  reported <- tabulate(onset + 1L, cutoff + 1L)
  onset[runif(length(onset)) < 0.4] <- NA
  list(
    onset = onset,
    report = report,
    cutoff = cutoff,
    delay = delay,
    reported = reported,
    eventual = eventual
  )
}

# Full weeks ending on the cut-off.
weeks_to_cutoff <- function(x) {
  n <- length(x) %/% 7
  tail_days <- x[(length(x) - 7 * n + 1):length(x)]
  colSums(matrix(tail_days, nrow = 7))
}

# Per-day delay pmfs from per-period estimates.
by_day <- function(res) {
  period <- findInterval(seq(0, res$cutoff), res$delay_period_start)
  res$em_delay_by_period[period, , drop = FALSE]
}

test_that(".delay_periods counts periods back from the cut-off", {
  expect_identical(.delay_periods(9), integer(10))
  expect_identical(
    .delay_periods(9, 4),
    c(0L, 0L, 1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L)
  )
  expect_identical(.delay_periods(9, 10), integer(10))
  expect_error(.delay_periods(9, 0), "delay_period")
  expect_error(.delay_periods(9, 2.5), "delay_period")
})

test_that("per-period EM beats static EM when delays fall", {
  imp <- list(static = 0, period = 0)
  now <- list(static = 0, period = 0)
  for (seed in 1:3) {
    ll <- simulate_falling(seed)
    for (variant in names(imp)) {
      period <- if (variant == "period") 14 else NULL
      res <- impute_linelist(
        ll$onset,
        ll$report,
        cutoff = ll$cutoff,
        delay_period = period
      )
      imputed <- weeks_to_cutoff(res$imputed$em)
      err <- abs(imputed - weeks_to_cutoff(ll$reported))
      imp[[variant]] <- imp[[variant]] + sum(err) / sum(ll$reported)
      nowcast <- correct_truncation(
        res$imputed$em,
        by_day(res),
        res$cutoff,
        min_cover = 1e-3
      )
      rel <- utils::tail(
        weeks_to_cutoff(nowcast) / weeks_to_cutoff(ll$eventual) - 1,
        4
      )
      now[[variant]] <- now[[variant]] + mean(abs(rel))
    }
  }
  expect_lt(imp$period, imp$static)
  expect_lt(now$period, 0.75 * now$static)
})

test_that("per-period delays follow the trend", {
  ll <- simulate_falling(1)
  res <- impute_linelist(
    ll$onset,
    ll$report,
    cutoff = ll$cutoff,
    delay_period = 28
  )
  means <- as.vector(res$em_delay_by_period %*% seq(0, 60))
  expect_identical(dim(res$em_delay_by_period), c(5L, 61L))
  expect_equal(rowSums(res$em_delay_by_period), rep(1, 5))
  expect_identical(res$delay_period_start, c(0L, 9L, 37L, 65L, 93L))
  expect_gt(means[1], means[5] + 2)
})

test_that("the default is one static delay and one period matches it", {
  ll <- simulate_falling(2, cutoff = 80)
  res <- impute_linelist(ll$onset, ll$report, cutoff = 80)
  expect_identical(res$em_delay_by_period, matrix(res$em_delay, nrow = 1))
  expect_identical(res$delay_period_start, 0L)
  one <- impute_linelist(ll$onset, ll$report, cutoff = 80, delay_period = 81)
  expect_identical(one$imputed$em, res$imputed$em)
  expect_identical(one$em_delay, res$em_delay)
})

test_that("a period with no cases takes the previous period's delay", {
  onset <- rep(c(0:13, 28:41), each = 5)
  delays <- c(rep(c(0, 1, 2, 3, 5), 14), rep(c(2, 4, 6, 7, 9), 14))
  res <- impute_linelist(
    onset,
    onset + delays,
    cutoff = 55,
    delay_period = 14
  )
  f <- res$em_delay_by_period
  expect_identical(res$delay_period_start, c(0L, 14L, 28L, 42L))
  expect_equal(f[2, ], f[1, ], tolerance = 1e-9)
  expect_equal(f[4, ], f[3, ], tolerance = 1e-9)
  expect_gt(max(abs(f[3, ] - f[1, ])), 1e-3)
})

test_that("a short line list works with periods", {
  res <- impute_linelist(
    c(0, 2, NA, 5, NA),
    c(3, 6, 7, 9, 12),
    cutoff = 12,
    delay_period = 4
  )
  expect_identical(dim(res$em_delay_by_period), c(4L, 61L))
  expect_equal(rowSums(res$em_delay_by_period), rep(1, 4))
  expect_equal(sum(res$imputed$em), 5)
})

test_that("a given delay is held fixed by em", {
  pmf <- delay_pmf(4, 3)
  res <- impute_linelist(
    c(0, 2, NA, 5, NA),
    c(3, 6, 7, 9, 12),
    delay = pmf
  )
  expect_equal(res$em_delay, pmf)
  expect_error(
    impute_linelist(c(0, NA), c(2, 3), delay = pmf, delay_period = 2),
    "not both"
  )
  expect_error(
    impute_linelist(c(0, NA), c(2, 3), delay_shrinkage = 0),
    "delay_shrinkage"
  )
})

test_that("a delay matrix is a plug-in and beats a static plug-in", {
  ll <- simulate_falling(3)
  res <- impute_linelist(
    ll$onset,
    ll$report,
    cutoff = ll$cutoff,
    delay = ll$delay
  )
  expect_identical(dim(res$em_delay_by_period), c(121L, 61L))
  expect_equal(res$em_delay_by_period[8, ], ll$delay[8, ] / sum(ll$delay[8, ]))
  # fwd renormalises over onset days 0..t, so every case is placed
  expect_equal(
    sum(res$imputed$fwd - res$complete),
    sum(res$missing_by_report)
  )
  err <- function(delay) {
    fit <- impute_linelist(
      ll$onset,
      ll$report,
      cutoff = ll$cutoff,
      delay = delay
    )
    nowcast <- correct_truncation(
      fit$imputed$em,
      delay,
      ll$cutoff,
      min_cover = 1e-3
    )
    rel <- utils::tail(
      weeks_to_cutoff(nowcast) / weeks_to_cutoff(ll$eventual) - 1,
      4
    )
    mean(abs(rel))
  }
  expect_lt(err(ll$delay), 0.5 * err(delay_pmf(8, 6.4)))
  expect_error(
    impute_linelist(
      ll$onset,
      ll$report,
      cutoff = ll$cutoff,
      delay = ll$delay[-1, ]
    ),
    "one row per onset day"
  )
})

test_that("correct_truncation takes a delay matrix", {
  delay <- rbind(c(0.5, 0.5), c(1, 0), c(0.25, 0.75))
  out <- correct_truncation(c(4, 4, 4), delay)
  expect_equal(out, c(4, 4, 16))
  expect_error(correct_truncation(c(4, 4), delay), "one row per day")
})
