"""
    round_half_up(x)

Round `x` to the nearest integer with halves rounded up, as
`floor(x + 0.5)`.
This matches JavaScript's `Math.round` and the Python package.
Julia's `round` rounds halves to even.

Throws `ArgumentError` if `x` is not finite.
"""
function round_half_up(x::Real)
    isfinite(x) || throw(ArgumentError("cannot round non-finite value $x"))
    return floor(Int, x + 0.5)
end

"""
    backsampling_allocate(missing_by_report, pmf)

Expected allocation of missing-onset cases under back-sampling.

Each case reported on day `t` is spread over onset days `t - d` with
weight `pmf[d + 1]`.
It is the expected value of the random draw, not a draw.
Element `i` of both `missing_by_report` and the result is day `i - 1`.
Mass that falls before day 0 is dropped, so the sum can be below
`sum(missing_by_report)`.
"""
function backsampling_allocate(missing_by_report::AbstractVector{<:Real},
        pmf::AbstractVector{<:Real})
    n_days = length(missing_by_report)
    allocation = zeros(n_days)
    for d in 0:(min(length(pmf), n_days) - 1)
        w = pmf[d + 1]
        for s in 0:(n_days - 1 - d)
            allocation[s + 1] += missing_by_report[s + d + 1] * w
        end
    end
    return allocation
end

"""
    mean_shift_allocate(missing_by_report, shift)

Move each missing-onset case back from its report day by `shift` days.

This is the mean shift method.
Element `i` of both `missing_by_report` and the result is day `i - 1`.
Cases moved before day 0 are dropped.
Throws `ArgumentError` if `shift` is negative.
"""
function mean_shift_allocate(missing_by_report::AbstractVector{<:Real},
        shift::Integer)
    shift >= 0 ||
        throw(ArgumentError("shift must be non-negative, got $shift"))
    n_days = length(missing_by_report)
    allocation = zeros(n_days)
    if shift < n_days
        allocation[1:(n_days - shift)] .= missing_by_report[(shift + 1):end]
    end
    return allocation
end

@doc raw"""
    em_fit(onset, delay, missing_report, cutoff; max_delay = 60,
           n_iter = 150)

Back-project missing-onset cases by EM with a static estimated delay.

`onset` and `delay` are the onset days and delays of complete cases.
`missing_report` holds the report days of cases with a missing onset.
All days are whole numbers in `0:cutoff`, and complete cases must have
`onset + delay <= cutoff` and `0 <= delay <= max_delay`.

Expected counts by onset day ``s`` and delay ``d`` are
``\lambda(s) f(d)``, observed only when ``s + d \le T`` with
``T`` = `cutoff`.
Starting values are the complete-case delay counts plus 0.5, normalised,
for ``f``, and the complete-case onset counts inflated by the overall
missing fraction, plus 0.1, for ``\lambda``.
Each iteration allocates the ``m(t)`` missing cases reported on day
``t`` as

```math
n_a(s, t - s) = m(t) \frac{\lambda(s) f(t - s)}
{\sum_{s'} \lambda(s') f(t - s')},
```

then, with ``n = n_c + n_a``, alternates three times between

```math
\lambda(s) = \frac{\sum_d n(s, d)}{F\big(\min(T - s, D)\big)},
\qquad
f(d) \propto \frac{\sum_s n(s, d)}{\sum_{s \le T - d} \lambda(s)},
```

where ``F`` is the delay CDF and ``D`` = `max_delay`.
``f`` is floored at ``10^{-12}`` and renormalised.

Returns a NamedTuple with `allocation` (expected missing-onset cases by
onset day), `onsets` (``\lambda``) and `delay` (``f`` on
`0:max_delay`).
"""
function em_fit(onset::AbstractVector{<:Integer},
        delay::AbstractVector{<:Integer},
        missing_report::AbstractVector{<:Integer}, cutoff::Integer;
        max_delay::Integer = DMAX, n_iter::Integer = 150)
    cutoff >= 0 ||
        throw(ArgumentError("cutoff must be non-negative, got $cutoff"))
    n_iter >= 1 ||
        throw(ArgumentError("n_iter must be at least 1, got $n_iter"))
    max_delay >= 0 ||
        throw(ArgumentError("max_delay must be non-negative, got $max_delay"))
    length(onset) == length(delay) || throw(ArgumentError(
        "complete onset days and delays must have the same length"))
    all(s -> 0 <= s <= cutoff, onset) ||
        throw(ArgumentError("complete onset days must lie in 0:cutoff"))
    all(d -> 0 <= d <= max_delay, delay) ||
        throw(ArgumentError("complete delays must lie in 0:$max_delay"))
    all(onset .+ delay .<= cutoff) || throw(ArgumentError(
        "complete cases must be reported by the cutoff"))
    all(t -> 0 <= t <= cutoff, missing_report) ||
        throw(ArgumentError("missing report days must lie in 0:cutoff"))

    T, D = Int(cutoff), Int(max_delay)
    n_days = T + 1
    n_complete = zeros(n_days, D + 1)
    for (s, d) in zip(onset, delay)
        n_complete[s + 1, d + 1] += 1.0
    end
    missing_by_report = zeros(n_days)
    for t in missing_report
        missing_by_report[t + 1] += 1.0
    end
    f = vec(sum(n_complete; dims = 1)) .+ 0.5
    f ./= sum(f)
    n_missing = sum(missing_by_report)
    n_cases = n_missing + sum(n_complete)
    missing_frac = n_cases > 0 ? n_missing / n_cases : 0.0
    λ = vec(sum(n_complete; dims = 2)) ./ max(1e-9, 1 - missing_frac) .+ 0.1
    allocated = zeros(n_days, D + 1)
    denom = zeros(n_days)
    days_left = [min(T - s, D) for s in 0:T]
    for _ in 1:n_iter
        # E-step: allocate missing cases over onset days by report day
        fill!(denom, 0.0)
        for d in 0:D, s in 0:(T - d)

            denom[s + d + 1] += λ[s + 1] * f[d + 1]
        end
        fill!(allocated, 0.0)
        for d in 0:D, s in 0:(T - d)

            t = s + d
            allocated[s + 1, d + 1] = λ[s + 1] * f[d + 1] *
                                      missing_by_report[t + 1] /
                                      max(denom[t + 1], 1e-300)
        end
        total = n_complete .+ allocated
        by_onset = vec(sum(total; dims = 2))
        by_delay = vec(sum(total; dims = 1))
        # M-step: right-truncated triangle, alternating updates
        for _ in 1:3
            cdf = cumsum(f)
            λ = by_onset ./ cdf[days_left .+ 1]
            cum_λ = cumsum(λ)
            for d in 0:D
                exposure = T - d >= 0 ? cum_λ[T - d + 1] : 0.0
                f[d + 1] = max(by_delay[d + 1] / max(exposure, 1e-300), 1e-12)
            end
            f ./= sum(f)
        end
    end
    return (allocation = vec(sum(allocated; dims = 2)), onsets = λ,
        delay = f)
end
