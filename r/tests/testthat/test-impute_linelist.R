onset <- c(0, 2, NA, 5, NA)
report <- c(3, 6, 7, 9, 12)

test_that("impute_linelist returns the documented structure", {
  res <- impute_linelist(onset, report, cutoff = 12)
  expect_named(
    res,
    c(
      "complete", "missing_by_report", "empirical_delay", "imputed",
      "em_delay", "dropped", "imputed_before_day0", "cutoff", "origin"
    )
  )
  expect_named(res$imputed, c("backsampling", "mean", "em"))
  expect_named(res$imputed_before_day0, c("backsampling", "mean"))
  for (m in res$imputed) expect_length(m, 13)
  expect_length(res$empirical_delay, 61)
  expect_length(res$em_delay, 61)
  expect_identical(res$dropped, 0L)
  expect_identical(res$cutoff, 12L)
  expect_null(res$origin)
})

test_that("complete, missing and empirical delays are counted", {
  res <- impute_linelist(onset, report, cutoff = 12)
  expected_complete <- numeric(13)
  expected_complete[c(1, 3, 6)] <- 1
  expect_identical(res$complete, expected_complete)
  expected_missing <- numeric(13)
  expected_missing[c(8, 13)] <- 1
  expect_identical(res$missing_by_report, expected_missing)
  # delays are 3, 4 and 4
  expect_equal(res$empirical_delay[4:5], c(1 / 3, 2 / 3))
  expect_equal(sum(res$empirical_delay), 1)
})

test_that("back-sampling spreads cases by the empirical delay", {
  res <- impute_linelist(onset, report, cutoff = 12)
  bs <- res$imputed$backsampling - res$complete
  # case on day 7 goes to days 4 (1/3) and 3 (2/3); day 12 to 9 and 8
  expected <- numeric(13)
  expected[c(5, 4, 10, 9)] <- c(1, 2, 1, 2) / 3
  expect_equal(bs, expected)
  expect_equal(sum(res$imputed$backsampling), 5)
})

test_that("mean shift rounds halves up", {
  # delays 1 and 2 give 1.5: half up is 2, so day 5 goes to day 3
  res <- impute_linelist(c(0, 0, NA), c(1, 2, 5))
  expect_identical(which(res$imputed$mean - res$complete > 0), 4L)
  # delays 2 and 3 give 2.5: half up is 3 (day 3), half even 2 (day 4)
  res <- impute_linelist(c(0, 0, NA), c(2, 3, 6))
  expect_identical(which(res$imputed$mean - res$complete > 0), 4L)
})

test_that("cases placed before day 0 are reported", {
  res <- impute_linelist(c(0, 0, NA), c(4, 4, 2))
  expect_identical(res$imputed_before_day0$mean, 1)
  expect_identical(res$imputed_before_day0$backsampling, 1)
  expect_equal(sum(res$imputed$mean), 2)
})

test_that("EM allocates every missing case", {
  res <- impute_linelist(onset, report, cutoff = 12)
  expect_equal(sum(res$imputed$em), 5)
  expect_equal(sum(res$em_delay), 1)
  expect_true(all(res$em_delay > 0))
})

test_that("fwd uses the given delay", {
  delay <- c(0.5, 0.5)
  res <- impute_linelist(onset, report, cutoff = 12, delay = delay * 4)
  expect_named(res$imputed, c("backsampling", "mean", "em", "fwd"))
  fwd <- res$imputed$fwd - res$complete
  expected <- numeric(13)
  expected[c(8, 7, 13, 12)] <- 0.5
  expect_equal(fwd, expected)
  expect_identical(res$imputed_before_day0$fwd, 0)
})

test_that("out of range cases are dropped", {
  res <- impute_linelist(
    onset = c(0, 5, -1, 0, NA, 1),
    report = c(1, 3, 2, 70, 20, 2),
    cutoff = 10, max_delay = 5
  )
  # onset after report, negative onset, report after cutoff (twice)
  expect_identical(res$dropped, 4L)
  expect_identical(sum(res$complete), 2)
  res <- impute_linelist(c(0, 0), c(1, 9), max_delay = 5)
  expect_identical(res$dropped, 1L)
  expect_length(res$empirical_delay, 6)
})

test_that("cutoff defaults to the last report day", {
  res <- impute_linelist(onset, report)
  expect_identical(res$cutoff, 12L)
  res <- impute_linelist(onset, report, cutoff = 8)
  expect_length(res$complete, 9)
  expect_identical(res$dropped, 2L)
})

test_that("Date input is converted relative to the earliest date", {
  day0 <- as.Date("2026-05-01")
  res_date <- impute_linelist(day0 + onset, day0 + report)
  res_num <- impute_linelist(onset, report)
  expect_identical(res_date$origin, day0)
  expect_identical(res_date$imputed, res_num$imputed)
  res_cut <- impute_linelist(day0 + onset, day0 + report, cutoff = day0 + 9)
  expect_identical(res_cut$cutoff, 9L)
  # an earlier report than any onset sets day 0
  res <- impute_linelist(day0 + c(1, NA), day0 + c(2, 0))
  expect_identical(res$origin, day0)
  # all-missing onsets given as NA are allowed with Date reports
  expect_error(
    impute_linelist(c(NA, NA), day0 + c(1, 2)),
    "no complete cases"
  )
})
