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

# EM back-projection with a static delay estimated jointly with the onset
# curve, allowing for right truncation at the cut-off. Follows em_fit() in
# the Python package. Inputs are integer onset days and delays of complete
# cases and report days of missing-onset cases, all already in range.
.em_fit <- function(onset_c, delay_c, report_m, cutoff, max_delay,
                    n_iter = 150) {
  n_days <- cutoff + 1L
  n_delay <- max_delay + 1L
  n_complete <- matrix(
    tabulate(onset_c * n_delay + delay_c + 1L, n_days * n_delay),
    nrow = n_days, ncol = n_delay, byrow = TRUE
  )
  missing <- tabulate(report_m + 1L, n_days)
  report_grid <- outer(seq(0, cutoff), seq(0, max_delay), "+")
  observed <- report_grid <= cutoff
  report_idx <- ifelse(observed, report_grid, 0L) + 1L
  delay <- colSums(n_complete) + 0.5
  delay <- delay / sum(delay)
  n_cases <- sum(missing) + sum(n_complete)
  missing_frac <- if (n_cases > 0) sum(missing) / n_cases else 0
  onsets <- rowSums(n_complete) / max(1e-9, 1 - missing_frac) + 0.1
  allocated <- matrix(0, n_days, n_delay)
  days_left <- pmin(cutoff - seq(0, cutoff), max_delay)
  exposure_idx <- cutoff - seq(0, max_delay) + 1L
  for (iter in seq_len(n_iter)) {
    expected <- outer(onsets, delay) * observed
    # every report day 0..cutoff occurs (delay 0), so rowsum covers all days
    denom <- as.vector(rowsum(as.vector(expected), as.vector(report_idx)))
    ratio <- missing / pmax(denom, 1e-300)
    allocated <- expected * ratio[report_idx] * observed
    total <- n_complete + allocated
    by_onset <- rowSums(total)
    by_delay <- colSums(total)
    for (step in 1:3) {
      delay_cdf <- cumsum(delay)
      onsets <- by_onset / delay_cdf[days_left + 1L]
      cum_onsets <- cumsum(onsets)
      exposure <- ifelse(
        exposure_idx >= 1L, cum_onsets[pmax(exposure_idx, 1L)], 0
      )
      delay <- by_delay / pmax(exposure, 1e-300)
      delay <- pmax(delay, 1e-12)
      delay <- delay / sum(delay)
    }
  }
  list(allocation = rowSums(allocated), onsets = onsets, delay = delay)
}
