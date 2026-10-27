test_that("delay_pmf has the right length and sums to 1", {
  pmf <- delay_pmf(8, 8)
  expect_length(pmf, 61)
  expect_equal(sum(pmf), 1, tolerance = 1e-12)
  expect_true(all(pmf >= 0))
  expect_length(delay_pmf(4, 2, dmax = 10), 11)
  expect_identical(delay_pmf(4, 2, dmax = 0), 1)
})

test_that("delay_pmf follows the double censored formula", {
  mean <- 6.6
  sd <- 6.6 * 0.92
  shape <- (mean / sd)^2
  scale <- sd^2 / mean
  big_g <- function(x) {
    x <- pmax(x, 0)
    x * pgamma(x, shape, scale = scale) -
      shape * scale * pgamma(x, shape + 1, scale = scale)
  }
  d <- 0:30
  raw <- big_g(d + 1) - 2 * big_g(d) + big_g(d - 1)
  expect_equal(delay_pmf(mean, sd, dmax = 30), raw / sum(raw))
})

test_that("delay_pmf mean is close to the continuous mean", {
  pmf <- delay_pmf(8, 4, dmax = 100)
  # a uniform primary event and a day-censored report cancel out
  expect_equal(sum(0:100 * pmf), 8, tolerance = 1e-3)
})

test_that("delay_pmf validates its inputs", {
  expect_error(delay_pmf(0, 1), "mean must be a positive finite number")
  expect_error(delay_pmf(-1, 1), "mean must be a positive")
  expect_error(delay_pmf(Inf, 1), "mean must be a positive")
  expect_error(delay_pmf(NA_real_, 1), "mean must be a positive")
  expect_error(delay_pmf(c(1, 2), 1), "mean must be a positive")
  expect_error(delay_pmf(1, 0), "sd must be a positive")
  expect_error(delay_pmf(1, "a"), "sd must be a positive")
  expect_error(delay_pmf(1, 1, dmax = -1), "dmax must be at least 0")
  expect_error(delay_pmf(1, 1, dmax = 1.5), "dmax must be a single whole")
})
