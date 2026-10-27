# Internal input checks.

.is_whole <- function(x) {
  is.finite(x) & x == round(x)
}

.check_scalar_whole <- function(x, what, min = -Inf) {
  if (!is.numeric(x) || length(x) != 1L || !.is_whole(x)) {
    stop(what, " must be a single whole number, got ",
      format(x)[1L],
      call. = FALSE
    )
  }
  if (x < min) {
    stop(what, " must be at least ", min, ", got ", x, call. = FALSE)
  }
  invisible(as.integer(x))
}

.check_positive <- function(x, what) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x <= 0) {
    stop(what, " must be a positive finite number, got ",
      format(x)[1L],
      call. = FALSE
    )
  }
  invisible(x)
}

.check_pmf <- function(x, what) {
  if (!is.numeric(x) || length(x) == 0L) {
    stop(what, " must be a non-empty numeric vector", call. = FALSE)
  }
  if (any(!is.finite(x)) || any(x < 0)) {
    stop(what, " must hold finite non-negative values", call. = FALSE)
  }
  if (sum(x) <= 0) {
    stop(what, " must have a positive sum", call. = FALSE)
  }
  x / sum(x)
}

.round_half_up <- function(x) {
  if (!is.finite(x)) {
    stop("cannot round non-finite value ", x, call. = FALSE)
  }
  as.integer(floor(x + 0.5))
}
