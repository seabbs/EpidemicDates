test_that("correct_truncation divides by the delay CDF", {
  delay <- c(0.5, 0.3, 0.2)
  counts <- c(10, 10, 10, 10)
  out <- correct_truncation(counts, delay)
  expect_equal(out, c(10, 10, 10 / 0.8, 10 / 0.5))
})

test_that("correct_truncation sets poorly covered days to NA", {
  delay <- c(0.01, 0.99)
  out <- correct_truncation(c(5, 5, 5), delay)
  expect_identical(is.na(out), c(FALSE, FALSE, TRUE))
  out <- correct_truncation(c(5, 5, 5), delay, min_cover = 0)
  expect_equal(out[3], 500)
  # day 1 is the cut-off, 1% reported; day 2 is after it
  out <- correct_truncation(c(5, 5, 5), delay, cutoff = 1)
  expect_identical(is.na(out), c(FALSE, TRUE, TRUE))
})

test_that("correct_truncation normalises the delay", {
  expect_equal(
    correct_truncation(c(1, 1), c(1, 1)),
    correct_truncation(c(1, 1), c(0.5, 0.5))
  )
})

test_that("correct_truncation validates its inputs", {
  expect_error(correct_truncation("a", 1), "counts must be numeric")
  expect_error(correct_truncation(1, -1), "non-negative")
  expect_error(correct_truncation(1, 1, cutoff = 0.5), "single whole")
  expect_error(correct_truncation(1, 1, min_cover = 2), "min_cover")
})
