# Internal imputation methods. Day vectors use index i for day i - 1.

# Expected allocation of back-sampling: spread each case reported on day t
# over onset days t - d with weight pmf[d + 1]. Mass before day 0 is lost.
.backsampling_allocate <- function(missing_by_report, pmf) {
  n_days <- length(missing_by_report)
  allocation <- numeric(n_days)
  for (delay in seq_len(min(length(pmf), n_days)) - 1L) {
    idx <- seq_len(n_days - delay)
    allocation[idx] <- allocation[idx] +
      missing_by_report[idx + delay] * pmf[delay + 1L]
  }
  allocation
}

# Move each case back by a fixed shift. Cases moved before day 0 are lost.
.mean_shift_allocate <- function(missing_by_report, shift) {
  if (shift < 0) {
    stop("shift must be non-negative, got ", shift, call. = FALSE)
  }
  n_days <- length(missing_by_report)
  allocation <- numeric(n_days)
  if (shift < n_days) {
    idx <- seq_len(n_days - shift)
    allocation[idx] <- missing_by_report[idx + shift]
  }
  allocation
}

# Period index (0-based) of each onset day 0..cutoff, for periods of
# delay_period days counted back from the cut-off. NULL gives one period.
.delay_periods <- function(cutoff, delay_period = NULL) {
  if (is.null(delay_period)) {
    return(integer(cutoff + 1L))
  }
  ok <- is.numeric(delay_period) &&
    length(delay_period) == 1L &&
    .is_whole(delay_period) &&
    delay_period >= 1
  if (!ok) {
    stop(
      "delay_period must be a whole number of days of at least 1, got ",
      format(delay_period)[1L],
      call. = FALSE
    )
  }
  n_periods <- ceiling((cutoff + 1) / delay_period)
  as.integer(n_periods - 1 - (cutoff - seq(0, cutoff)) %/% delay_period)
}

# Expected allocation with P(s | t) proportional to onsets[s] times the
# delay of onset day s at t - s, renormalised over onset days 0..t.
# delay_by_day has one row per onset day.
.posterior_allocate <- function(missing_by_report, onsets, delay_by_day) {
  n_days <- length(missing_by_report)
  n_delay <- ncol(delay_by_day)
  out <- numeric(n_days)
  for (t in which(missing_by_report > 0) - 1L) {
    d <- seq(0, min(n_delay - 1L, t))
    w <- onsets[t - d + 1L] * delay_by_day[cbind(t - d + 1L, d + 1L)]
    if (sum(w) <= 0) {
      next
    }
    out[t - d + 1L] <- out[t - d + 1L] + missing_by_report[t + 1L] * w / sum(w)
  }
  out
}

# Check a supplied delay: a pmf, or a matrix with one pmf per onset day
# 0..cutoff in its rows. Returns a matrix with each pmf on 0..max_delay,
# renormalised, with one row for a static delay.
.as_delay_matrix <- function(delay, cutoff, max_delay) {
  rows <- if (is.matrix(delay)) delay else matrix(delay, nrow = 1L)
  if (is.matrix(delay) && nrow(delay) != cutoff + 1L) {
    stop(
      "a delay matrix needs one row per onset day 0..cutoff (",
      cutoff + 1L,
      " rows), got ",
      nrow(delay),
      call. = FALSE
    )
  }
  out <- matrix(0, nrow(rows), max_delay + 1L)
  n <- min(ncol(rows), max_delay + 1L)
  for (k in seq_len(nrow(rows))) {
    row <- .check_pmf(rows[k, ], "delay")
    out[k, seq_len(n)] <- row[seq_len(n)]
    if (sum(out[k, ]) <= 0) {
      stop("delay must have positive mass on 0..", max_delay, call. = FALSE)
    }
    out[k, ] <- out[k, ] / sum(out[k, ])
  }
  out
}

# EM back-projection with the delay estimated jointly with the onset curve,
# allowing for right truncation at the cut-off. Follows em_fit() in the
# Python package. Inputs are integer onset days and delays of complete
# cases and report days of missing-onset cases, all already in range.
# `fixed` is NULL to estimate the delay, or a matrix of delay pmfs to hold
# fixed with one row per period. `period` gives the 0-based period of each
# onset day. With several periods and no fixed delay, each period's delay
# is shrunk towards the previous period's, the first towards the pooled
# delay, with strength `shrinkage`.
.em_fit <- function(
  onset_c,
  delay_c,
  report_m,
  cutoff,
  max_delay,
  n_iter = 150,
  fixed = NULL,
  period = NULL,
  shrinkage = 10
) {
  n_days <- cutoff + 1L
  n_delay <- max_delay + 1L
  if (is.null(period)) {
    period <- integer(n_days)
  }
  row_of_day <- period + 1L
  n_periods <- row_of_day[n_days]
  starts <- which(c(TRUE, diff(period) != 0L))
  ends <- c(starts[-1L] - 1L, n_days)
  n_complete <- matrix(
    tabulate(onset_c * n_delay + delay_c + 1L, n_days * n_delay),
    nrow = n_days,
    ncol = n_delay,
    byrow = TRUE
  )
  missing <- tabulate(report_m + 1L, n_days)
  report_grid <- outer(seq(0, cutoff), seq(0, max_delay), "+")
  observed <- report_grid <= cutoff
  report_idx <- ifelse(observed, report_grid, 0L) + 1L
  fix <- !is.null(fixed)
  if (fix) {
    by_period <- fixed
  } else {
    delay <- colSums(n_complete) + 0.5
    delay <- delay / sum(delay)
    by_period <- matrix(delay, n_periods, n_delay, byrow = TRUE)
  }
  pooled <- by_period[1L, ]
  n_cases <- sum(missing) + sum(n_complete)
  missing_frac <- if (n_cases > 0) sum(missing) / n_cases else 0
  onsets <- rowSums(n_complete) / max(1e-9, 1 - missing_frac) + 0.1
  allocated <- matrix(0, n_days, n_delay)
  days_left <- pmin(cutoff - seq(0, cutoff), max_delay)
  exposure_idx <- cutoff - seq(0, max_delay) + 1L
  for (iter in seq_len(n_iter)) {
    expected <- onsets * by_period[row_of_day, , drop = FALSE] * observed
    # every report day 0..cutoff occurs (delay 0), so rowsum covers all days
    denom <- as.vector(rowsum(as.vector(expected), as.vector(report_idx)))
    ratio <- missing / pmax(denom, 1e-300)
    allocated <- expected * ratio[report_idx] * observed
    total <- n_complete + allocated
    by_onset <- rowSums(total)
    by_delay <- colSums(total)
    for (step in seq_len(if (fix) 1L else 3L)) {
      cdf <- by_period
      for (k in seq_len(nrow(by_period))) {
        cdf[k, ] <- cumsum(by_period[k, ])
      }
      onsets <- by_onset / pmax(cdf[cbind(row_of_day, days_left + 1L)], 1e-12)
      if (fix) {
        next
      }
      cum_onsets <- cumsum(onsets)
      exposure <- ifelse(
        exposure_idx >= 1L,
        cum_onsets[pmax(exposure_idx, 1L)],
        0
      )
      pooled <- by_delay / pmax(exposure, 1e-300)
      pooled <- pmax(pooled, 1e-12)
      pooled <- pooled / sum(pooled)
      if (n_periods == 1L) {
        by_period <- matrix(pooled, nrow = 1L)
        next
      }
      by_period <- .shrink_periods(
        rowsum(total, row_of_day),
        pooled,
        cum_onsets,
        starts,
        ends,
        cutoff,
        shrinkage
      )
    }
  }
  if (fix) {
    pooled <- if (nrow(by_period) == 1L) {
      by_period[1L, ]
    } else {
      weights <- pmax(onsets, 0)
      if (sum(weights) <= 0) {
        weights <- rep(1, n_days)
      }
      colSums(weights * by_period[row_of_day, , drop = FALSE]) / sum(weights)
    }
  }
  list(
    allocation = rowSums(allocated),
    onsets = onsets,
    delay = pooled,
    delay_by_period = unname(by_period),
    period_start = starts - 1L
  )
}

# Shrink each period's right-truncated delay estimate towards the previous
# period's delay, and the first towards the pooled delay:
# f_k(d) is proportional to (N_k(d) + shrinkage * h_k(d)) /
# (E_k(d) + shrinkage), where E_k(d) sums the onset curve over days of
# period k up to cutoff - d. `starts` and `ends` are 1-based day indices.
.shrink_periods <- function(
  counts,
  pooled,
  cum_onsets,
  starts,
  ends,
  cutoff,
  shrinkage
) {
  out <- counts
  last_seen <- cutoff - seq(0, length(pooled) - 1L) + 1L
  target <- pooled
  for (k in seq_len(nrow(counts))) {
    before <- if (starts[k] > 1L) cum_onsets[starts[k] - 1L] else 0
    upto <- pmin(last_seen, ends[k])
    exposure <- ifelse(
      upto >= starts[k],
      cum_onsets[pmax(upto, 1L)] - before,
      0
    )
    exposure <- pmax(exposure, 0)
    row <- (counts[k, ] + shrinkage * target) / (exposure + shrinkage)
    row <- pmax(row, 1e-12)
    target <- row / sum(row)
    out[k, ] <- target
  }
  out
}
