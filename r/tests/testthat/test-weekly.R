test_that("weekly sums days to weeks", {
  expect_identical(weekly(rep(1, 14)), c(7, 7))
  expect_identical(weekly(1:15), c(28L, 77L, 15L))
  expect_identical(weekly(1:15, start = 1), c(35L, 84L))
  expect_identical(weekly(1:3, start = 3), numeric(0))
  expect_identical(weekly(c(1, NA, 1)), NA_real_)
})

test_that("weekly validates its inputs", {
  expect_error(weekly("a"), "daily must be numeric")
  expect_error(weekly(1:3, start = -1), "start must be at least")
})
