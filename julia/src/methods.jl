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
function backsampling_allocate(
        missing_by_report::AbstractVector{<:Real},
        pmf::AbstractVector{<:Real}
    )
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
function mean_shift_allocate(
        missing_by_report::AbstractVector{<:Real},
        shift::Integer
    )
    shift >= 0 ||
        throw(ArgumentError("shift must be non-negative, got $shift"))
    n_days = length(missing_by_report)
    allocation = zeros(n_days)
    if shift < n_days
        allocation[1:(n_days - shift)] .= missing_by_report[(shift + 1):end]
    end
    return allocation
end

"""
    delay_periods(cutoff, delay_period = nothing)

Period index (0, 1, ...) of each onset day `0:cutoff`, for periods of
`delay_period` days counted back from the cut-off.

The latest period is always full length and the first holds the days
left over.
`nothing` gives one period.
Throws `ArgumentError` if `delay_period` is not a whole number of at
least 1.

# Examples
```jldoctest
julia> EpidemicDates.delay_periods(9, 4)
10-element Vector{Int64}:
 0
 0
 1
 1
 1
 1
 2
 2
 2
 2
```
"""
function delay_periods(cutoff::Integer, delay_period = nothing)
    delay_period === nothing && return zeros(Int, cutoff + 1)
    (
        delay_period isa Real && !(delay_period isa Bool) &&
            _is_whole(delay_period) && delay_period >= 1
    ) ||
        throw(
        ArgumentError(
            "delay_period must be a whole number of days " *
                "of at least 1, got $delay_period"
        )
    )
    p = Int(delay_period)
    n_periods = cld(cutoff + 1, p)
    return [n_periods - 1 - fld(cutoff - s, p) for s in 0:cutoff]
end

# First onset day of each period.
function _period_starts(period::AbstractVector{<:Integer})
    return [
        s - 1
            for s in eachindex(period)
            if s == 1 || period[s] != period[s - 1]
    ]
end

"""
    posterior_allocate(missing_by_report, onsets, delay_by_day)

Expected allocation of missing-onset cases with
`P(s | t) ∝ onsets[s + 1] * delay_by_day[s + 1, t - s + 1]`, renormalised
over onset days `0:t`.
With constant onsets this is the forward draw with a delay that changes
over time.
`delay_by_day` has one row per onset day.
"""
function posterior_allocate(
        missing_by_report::AbstractVector{<:Real},
        onsets::AbstractVector{<:Real}, delay_by_day::AbstractMatrix{<:Real}
    )
    n_days = length(missing_by_report)
    n_delay = size(delay_by_day, 2)
    out = zeros(n_days)
    for t in 0:(n_days - 1)
        m = missing_by_report[t + 1]
        m > 0 || continue
        ds = 0:min(n_delay - 1, t)
        w = [onsets[t - d + 1] * delay_by_day[t - d + 1, d + 1] for d in ds]
        total = sum(w)
        total > 0 || continue
        for (i, d) in enumerate(ds)
            out[t - d + 1] += m * w[i] / total
        end
    end
    return out
end

@doc raw"""
    em_fit(onset, delay, missing_report, cutoff; max_delay = 60,
           n_iter = 150, fixed = nothing, period = nothing,
           delay_shrinkage = 10.0)

Back-project missing-onset cases by EM, estimating the delay or holding
a given one fixed.

`onset` and `delay` are the onset days and delays of complete cases.
`missing_report` holds the report days of cases with a missing onset.
All days are whole numbers in `0:cutoff`, and complete cases must have
`onset + delay <= cutoff` and `0 <= delay <= max_delay`.

`period` gives the period (0, 1, ...) of each onset day, from
`EpidemicDates.delay_periods`; by default there is
one period.
Expected counts by onset day ``s`` and delay ``d`` are
``\lambda(s) f_{k(s)}(d)``, observed only when ``s + d \le T`` with
``T`` = `cutoff` and ``k(s)`` the period of day ``s``.
Starting values are the complete-case delay counts plus 0.5, normalised,
for every ``f_k``, and the complete-case onset counts inflated by the
overall missing fraction, plus 0.1, for ``\lambda``.
Each iteration allocates the ``m(t)`` missing cases reported on day
``t`` as

```math
n_a(s, t - s) = m(t) \frac{\lambda(s) f_{k(s)}(t - s)}
{\sum_{s'} \lambda(s') f_{k(s')}(t - s')},
```

then, with ``n = n_c + n_a``, alternates three times between

```math
\lambda(s) = \frac{\sum_d n(s, d)}{F_{k(s)}\big(\min(T - s, D)\big)},
\qquad
f(d) \propto \frac{\sum_s n(s, d)}{\sum_{s \le T - d} \lambda(s)},
```

where ``F_k`` is the CDF of ``f_k`` and ``D`` = `max_delay`.
``f`` is the pooled delay, floored at ``10^{-12}`` and renormalised.
With one period ``f_0 = f``.
With several, in order ``k = 0, 1, \dots``,

```math
f_k(d) \propto \frac{N_k(d) + \kappa h_k(d)}{E_k(d) + \kappa},
\qquad h_0 = f, \quad h_k = f_{k-1},
```

with ``N_k(d)`` the cases with onset in period ``k`` and delay ``d``,
``E_k(d) = \sum_{s \in k, s \le T - d} \lambda(s)`` and ``\kappa`` =
`delay_shrinkage`.
If `fixed` is a matrix of delay pmfs, one row per period, the delays are
held fixed and only ``\lambda`` is updated.

Returns a NamedTuple with `allocation` (expected missing-onset cases by
onset day), `onsets` (``\lambda``), `delay` (the pooled ``f`` on
`0:max_delay`, or the onset-weighted mean of a fixed delay matrix),
`delay_by_period` (one row per period) and `period_start`.
"""
function em_fit(
        onset::AbstractVector{<:Integer},
        delay::AbstractVector{<:Integer},
        missing_report::AbstractVector{<:Integer}, cutoff::Integer;
        max_delay::Integer = DMAX, n_iter::Integer = 150,
        fixed::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        period::Union{Nothing, AbstractVector{<:Integer}} = nothing,
        delay_shrinkage::Real = 10.0
    )
    cutoff >= 0 ||
        throw(ArgumentError("cutoff must be non-negative, got $cutoff"))
    n_iter >= 1 ||
        throw(ArgumentError("n_iter must be at least 1, got $n_iter"))
    max_delay >= 0 ||
        throw(ArgumentError("max_delay must be non-negative, got $max_delay"))
    length(onset) == length(delay) || throw(
        ArgumentError(
            "complete onset days and delays must have the same length"
        )
    )
    all(s -> 0 <= s <= cutoff, onset) ||
        throw(ArgumentError("complete onset days must lie in 0:cutoff"))
    all(d -> 0 <= d <= max_delay, delay) ||
        throw(ArgumentError("complete delays must lie in 0:$max_delay"))
    all(onset .+ delay .<= cutoff) || throw(
        ArgumentError(
            "complete cases must be reported by the cutoff"
        )
    )
    all(t -> 0 <= t <= cutoff, missing_report) ||
        throw(ArgumentError("missing report days must lie in 0:cutoff"))

    T, D = Int(cutoff), Int(max_delay)
    n_days = T + 1
    per = period === nothing ? zeros(Int, n_days) : collect(Int, period)
    length(per) == n_days || throw(
        ArgumentError(
            "period must have one entry per onset day 0:cutoff"
        )
    )
    rows = per .+ 1
    n_periods = rows[end]
    starts = _period_starts(per)
    ends = vcat(starts[2:end] .- 1, T)
    κ = Float64(delay_shrinkage)
    n_complete = zeros(n_days, D + 1)
    for (s, d) in zip(onset, delay)
        n_complete[s + 1, d + 1] += 1.0
    end
    missing_by_report = zeros(n_days)
    for t in missing_report
        missing_by_report[t + 1] += 1.0
    end
    fix = fixed !== nothing
    if fix
        size(fixed) == (n_periods, D + 1) || throw(
            ArgumentError(
                "fixed must have one row per period and max_delay + 1 columns"
            )
        )
        by_period = Matrix{Float64}(fixed)
    else
        f0 = vec(sum(n_complete; dims = 1)) .+ 0.5
        f0 ./= sum(f0)
        by_period = repeat(permutedims(f0), n_periods)
    end
    f = by_period[1, :]
    n_missing = sum(missing_by_report)
    n_cases = n_missing + sum(n_complete)
    missing_frac = n_cases > 0 ? n_missing / n_cases : 0.0
    λ = vec(sum(n_complete; dims = 2)) ./ max(1.0e-9, 1 - missing_frac) .+ 0.1
    allocated = zeros(n_days, D + 1)
    denom = zeros(n_days)
    days_left = [min(T - s, D) for s in 0:T]
    cdf = similar(by_period)
    for _ in 1:n_iter
        # E-step: allocate missing cases over onset days by report day
        fill!(denom, 0.0)
        for d in 0:D, s in 0:(T - d)

            denom[s + d + 1] += λ[s + 1] * by_period[rows[s + 1], d + 1]
        end
        fill!(allocated, 0.0)
        for d in 0:D, s in 0:(T - d)

            t = s + d
            allocated[s + 1, d + 1] = λ[s + 1] *
                by_period[rows[s + 1], d + 1] *
                missing_by_report[t + 1] /
                max(denom[t + 1], 1.0e-300)
        end
        total = n_complete .+ allocated
        by_onset = vec(sum(total; dims = 2))
        by_delay = vec(sum(total; dims = 1))
        # M-step: right-truncated triangle, alternating updates
        for _ in 1:(fix ? 1 : 3)
            for k in 1:n_periods
                cdf[k, :] .= cumsum(by_period[k, :])
            end
            λ = [
                by_onset[s + 1] /
                    max(cdf[rows[s + 1], days_left[s + 1] + 1], 1.0e-12)
                    for s in 0:T
            ]
            fix && continue
            cum_λ = cumsum(λ)
            for d in 0:D
                exposure = T - d >= 0 ? cum_λ[T - d + 1] : 0.0
                f[d + 1] = max(by_delay[d + 1] / max(exposure, 1.0e-300), 1.0e-12)
            end
            f ./= sum(f)
            if n_periods == 1
                by_period[1, :] .= f
                continue
            end
            _shrink_periods!(
                by_period, total, f, cum_λ, rows, starts, ends,
                T, κ
            )
        end
    end
    if fix
        if n_periods == 1
            f = by_period[1, :]
        else
            w = max.(λ, 0.0)
            sum(w) > 0 || (w = ones(n_days))
            f = vec(sum(w .* by_period[rows, :]; dims = 1)) ./ sum(w)
        end
    end
    return (
        allocation = vec(sum(allocated; dims = 2)), onsets = λ,
        delay = f, delay_by_period = by_period, period_start = starts,
    )
end

# Shrink each period's right-truncated delay estimate towards the previous
# period's delay, and the first towards the pooled delay `f`, in place.
function _shrink_periods!(
        by_period, total, f, cum_λ, rows, starts, ends,
        T, κ
    )
    n_periods, n_delay = size(by_period)
    counts = zeros(n_periods, n_delay)
    for s in eachindex(rows), d in 1:n_delay

        counts[rows[s], d] += total[s, d]
    end
    target = f
    for k in 1:n_periods
        s0, s1 = starts[k], ends[k]
        before = s0 > 0 ? cum_λ[s0] : 0.0
        row = zeros(n_delay)
        for d in 0:(n_delay - 1)
            upto = min(T - d, s1)
            exposure = upto >= s0 ? max(cum_λ[upto + 1] - before, 0.0) : 0.0
            row[d + 1] = max(
                (counts[k, d + 1] + κ * target[d + 1]) /
                    (exposure + κ), 1.0e-12
            )
        end
        row ./= sum(row)
        by_period[k, :] .= row
        target = row
    end
    return by_period
end
