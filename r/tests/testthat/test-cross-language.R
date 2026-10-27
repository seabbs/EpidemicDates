# Compares with the reference outputs in tests/fixtures/cross-language.json at
# the repository root, written by the Python package. Skipped if the file is
# not found, for example in R CMD check, which copies the package elsewhere.

find_fixture <- function() {
  env <- Sys.getenv("EPIDEMICDATES_FIXTURES", "")
  candidates <- c(
    if (nzchar(env)) env,
    testthat::test_path("../../../tests/fixtures/cross-language.json")
  )
  dir <- normalizePath(testthat::test_path("."), mustWork = FALSE)
  for (i in 1:6) {
    candidates <- c(
      candidates,
      file.path(dir, "tests", "fixtures", "cross-language.json")
    )
    dir <- dirname(dir)
  }
  found <- candidates[file.exists(candidates)]
  if (length(found) == 0L) NULL else found[[1]]
}

read_fixture <- function() {
  testthat::skip_if_not_installed("jsonlite")
  path <- find_fixture()
  testthat::skip_if(is.null(path), "cross-language fixture not found")
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

test_that("delay_pmf matches the shared fixtures", {
  fx <- read_fixture()
  pmfs <- fx$delay_pmfs
  expect_gte(nrow(pmfs), 1)
  for (i in seq_len(nrow(pmfs))) {
    expect_equal(
      delay_pmf(pmfs$mean[i], pmfs$mean[i] * pmfs$cv[i], pmfs$dmax[i]),
      pmfs$pmf[[i]],
      tolerance = 1e-10
    )
  }
})

test_that("impute_linelist matches the shared fixtures", {
  fx <- read_fixture()
  ll <- fx$linelist
  res <- impute_linelist(
    ll$onset,
    ll$report,
    cutoff = ll$cutoff,
    max_delay = ll$max_delay
  )
  expect_equal(res$empirical_delay, ll$empirical_delay, tolerance = 1e-12)
  # back-sampling and mean must match exactly
  expect_equal(
    res$imputed$backsampling,
    ll$imputed$backsampling,
    tolerance = 1e-12
  )
  expect_equal(res$imputed$mean, ll$imputed$mean, tolerance = 1e-12)
  # em within 2% on weeks with at least 50 cases
  em <- weekly(res$imputed$em)
  ref <- weekly(ll$imputed$em)
  big <- ref >= 50
  expect_gte(sum(big), 1)
  expect_true(all(abs(em[big] - ref[big]) / ref[big] <= 0.02))
})

expect_weekly_close <- function(x, ref) {
  em <- weekly(x)
  ref <- weekly(ref)
  big <- ref >= 50
  expect_gte(sum(big), 1)
  expect_true(all(abs(em[big] - ref[big]) / ref[big] <= 0.02))
}

test_that("per-period EM matches the shared fixtures", {
  fx <- read_fixture()
  ll <- fx$linelist_periods
  res <- impute_linelist(
    ll$onset,
    ll$report,
    cutoff = ll$cutoff,
    max_delay = ll$max_delay,
    delay_period = ll$delay_period,
    delay_shrinkage = ll$delay_shrinkage
  )
  expect_identical(res$delay_period_start, as.integer(ll$delay_period_start))
  expect_equal(
    res$imputed$backsampling,
    ll$imputed$backsampling,
    tolerance = 1e-12
  )
  expect_equal(res$imputed$mean, ll$imputed$mean, tolerance = 1e-12)
  expect_weekly_close(res$imputed$em, ll$imputed$em)
})

test_that("a supplied delay matrix matches the shared fixtures", {
  fx <- read_fixture()
  ll <- fx$linelist_delay_matrix
  res <- impute_linelist(
    ll$onset,
    ll$report,
    cutoff = ll$cutoff,
    max_delay = ll$max_delay,
    delay = ll$delay
  )
  expect_equal(
    res$imputed$backsampling,
    ll$imputed$backsampling,
    tolerance = 1e-12
  )
  expect_equal(res$imputed$mean, ll$imputed$mean, tolerance = 1e-12)
  expect_equal(res$imputed$fwd, ll$imputed$forward, tolerance = 1e-12)
  expect_weekly_close(res$imputed$em, ll$imputed$em)
})
