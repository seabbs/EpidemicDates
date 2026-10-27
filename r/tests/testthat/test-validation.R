test_that("impute_linelist rejects bad line lists", {
  expect_error(impute_linelist(c(0, 1), 1), "same length")
  expect_error(impute_linelist(numeric(0), numeric(0)), "empty")
  expect_error(impute_linelist(c(0, 1), c(1, NA)), "report day is missing")
  expect_error(impute_linelist(c(0, 1), c(1, Inf)), "report day is missing")
  expect_error(impute_linelist(c(0, Inf), c(1, 2)), "must not be infinite")
  expect_error(impute_linelist(c(0, 1), c(1, 2.5)), "report days must be")
  expect_error(impute_linelist(c(0.5, 1), c(1, 2)), "onset days must be")
  expect_error(impute_linelist(c(NA, NA), c(1, 2)), "no complete cases")
  expect_error(impute_linelist(c(3, 3), c(1, 2)), "no complete cases")
  expect_error(impute_linelist("a", 1), "onset must be numeric")
  expect_error(impute_linelist(0, "1"), "report must be numeric")
})

test_that("impute_linelist rejects bad options", {
  expect_error(impute_linelist(0, 1, cutoff = -1), "cutoff must be at least")
  expect_error(impute_linelist(0, 1, cutoff = 1.5), "cutoff must be a single")
  expect_error(impute_linelist(0, 1, n_iter = 0), "n_iter must be at least")
  expect_error(impute_linelist(0, 1, max_delay = -1), "max_delay must be")
  expect_error(impute_linelist(0, 1, delay = c(-1, 2)), "non-negative")
  expect_error(impute_linelist(0, 1, delay = c(0, 0)), "positive sum")
  expect_error(impute_linelist(0, 1, delay = "a"), "numeric vector")
})

test_that("impute_linelist rejects mixed Date and numeric input", {
  day0 <- as.Date("2026-05-01")
  expect_error(impute_linelist(day0, 1), "both be Date")
  expect_error(impute_linelist(0, day0), "both be Date")
  expect_error(impute_linelist(0, 1, cutoff = day0), "only be a Date")
  expect_error(
    impute_linelist(day0, day0 + 1, cutoff = 1),
    "cutoff must be a Date"
  )
  expect_error(
    impute_linelist(as.Date(NA), as.Date(NA)),
    "missing for every case"
  )
})
