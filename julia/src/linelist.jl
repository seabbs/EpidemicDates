_is_missing_day(x) = ismissing(x) || isnothing(x) || (x isa Real && isnan(x))

_is_whole(x::Real) = isfinite(x) && x == round(x)

# Convert date vectors to whole days since the earliest date.
function _dates_to_days(
        onset::AbstractVector, report::AbstractVector,
        cutoff
    )
    known = [x for x in onset if !_is_missing_day(x)]
    all(x -> x isa Date, known) || throw(
        ArgumentError(
            "onset must hold dates or missing when report holds dates"
        )
    )
    reports = [x for x in report if !_is_missing_day(x)]
    all(x -> x isa Date, reports) ||
        throw(ArgumentError("report must hold only dates"))
    day0 = minimum(vcat(Date[x for x in reports], Date[x for x in known]))
    as_day(x) = _is_missing_day(x) ? missing : Dates.value(x - day0)
    cut = cutoff isa Date ? Dates.value(cutoff - day0) : cutoff
    return map(as_day, onset), map(as_day, report), cut, day0
end

# Check a supplied delay and return it as a matrix of pmfs on
# 0:max_delay, one row for a static delay or one per onset day 0:T.
function _fixed_delay(delay, T::Integer, max_delay::Integer)
    if delay isa AbstractMatrix
        size(delay, 1) == T + 1 || throw(
            ArgumentError(
                "a delay matrix needs one row per onset day 0:cutoff " *
                    "($(T + 1) rows), got $(size(delay, 1))"
            )
        )
        rows = [_check_delay(view(delay, i, :)) for i in axes(delay, 1)]
    else
        rows = [_check_delay(delay)]
    end
    out = zeros(length(rows), max_delay + 1)
    for (k, row) in enumerate(rows)
        n = min(length(row), max_delay + 1)
        out[k, 1:n] .= row[1:n]
        sum(out[k, :]) > 0 || throw(
            ArgumentError(
                "delay must have positive mass on 0:$max_delay"
            )
        )
        out[k, :] ./= sum(out[k, :])
    end
    return out
end

function _check_delay(delay)
    delay === nothing && return nothing
    v = collect(Float64, delay)
    isempty(v) && throw(ArgumentError("delay must not be empty"))
    all(x -> isfinite(x) && x >= 0, v) || throw(
        ArgumentError(
            "delay must hold non-negative finite probabilities"
        )
    )
    sum(v) > 0 || throw(ArgumentError("delay must have a positive sum"))
    return v ./ sum(v)
end

"""
    impute_linelist(onset, report; cutoff = nothing, delay = nothing,
                    max_delay = 60, n_iter = 150, delay_period = nothing,
                    delay_shrinkage = 10.0)

Impute missing onsets in a line list with each method.

Days are whole numbers, for example days since the first report.
`onset` holds the onset day of each case, with `missing` (or `NaN`)
where it is unknown.
`report` holds the report day of each case and must have no missing
values.
`onset` and `report` can instead be `Date` vectors.
Days are then counted from the earliest date in either vector, and
`cutoff` can be a `Date`.

Daily outputs have length `T + 1`, where `T` is the cut-off, and element
`i` is onset day `i - 1`.
They are expected counts, so they need not be whole numbers.
Sum them to weeks with [`weekly`](@ref) before comparing methods.

# Methods
- `backsampling` spreads each missing case over report day minus each
  delay, weighted by `empirical_delay`.
  It is the expected allocation of the random draw, not a draw.
- `mean` shifts each missing case back by the mean of `empirical_delay`,
  rounded with halves up (as JavaScript's `Math.round`).
- `em` is EM back-projection (see `EpidemicDates.em_fit`).
  By default it estimates a static delay jointly with the onset curve,
  allowing for right truncation at the cut-off.
  With `delay_period` it estimates one delay per period of onset days,
  each shrunk towards the previous period's delay.
  With `delay` it holds that delay fixed (a plug-in) and only estimates
  the onset curve.
- `fwd`, only when `delay` is given, back-samples from that delay
  instead of the empirical one.
  With a delay matrix it uses `P(s | t) ∝ f_s(t - s)` over onset days
  `0:t`.

All methods assume onsets are missing at random given the report day.
If missingness depends on the delay, all are biased.

# Arguments
- `onset`: onset day per case, `missing` where unknown.
- `report`: report day per case.
- `cutoff`: data cut-off day `T`, by default the last report day.
  Must be a non-negative whole number.
  Cases reported before day 0 or after `T` are dropped.
- `delay`: optional forward delay pmf on delays `0, 1, ...`, for
  example from [`delay_pmf`](@ref), or a matrix with one such pmf per
  onset day `0:T` in its rows, for a delay that changes over time.
  It is normalised before use and used as given.
  Cannot be combined with `delay_period`.
- `max_delay`: longest delay kept.
  Complete cases with a longer delay are dropped.
- `n_iter`: number of EM iterations. Must be at least 1.
- `delay_period`: length in days of the periods of onset days that get
  their own EM delay, counted back from the cut-off.
  `nothing` (the default) estimates one static delay.
- `delay_shrinkage`: strength, in expected cases, of the shrinkage of
  each period's delay towards the previous period's delay.
  Must be positive.

# Returns
A NamedTuple with:
- `complete`: onset counts of cases with a known onset;
- `missing_by_report`: missing-onset cases by report day;
- `empirical_delay`: pooled delay pmf on `0:max_delay` of complete cases;
- `imputed`: a NamedTuple of daily onset counts (complete plus imputed)
  for `backsampling`, `mean`, `em` and, with `delay`, `fwd`;
- `em_delay`: the delay pmf on `0:max_delay` used by EM: the static
  estimate, the pooled estimate with `delay_period`, or the given
  `delay` (its onset-weighted mean for a matrix);
- `em_delay_by_period`: a matrix with the delay pmf of each period in
  its rows, one row without `delay_period` and one row per onset day
  for a delay matrix;
- `delay_period_start`: the first onset day of each period;
- `imputed_before_day0`: expected missing-onset cases that
  `backsampling`, `mean` and `fwd` place before day 0, which are not in
  `imputed`;
- `dropped`: the number of cases dropped;
- `cutoff`: the cut-off used;
- `day0`: the date of day 0 for `Date` input, otherwise `nothing`.

Complete cases with a negative onset, an onset after report or a delay
over `max_delay` are dropped, as are cases reported outside `0:T`.

# Throws
`ArgumentError` if the line list is empty, `onset` and `report` differ
in length, a report day is missing or infinite, an onset is infinite, a
day is not a whole number, `cutoff` is negative or not whole,
`max_delay` is negative, `n_iter` is below 1, `delay` is not a valid
pmf or matrix of pmfs, both `delay` and `delay_period` are given,
`delay_period` or `delay_shrinkage` is out of range, or no complete case
is left after dropping.

# Examples
```jldoctest
julia> res = impute_linelist([0, 2, missing, 5, missing], [3, 6, 7, 9, 12]);

julia> length(res.imputed.backsampling)
13

julia> res.imputed.mean
13-element Vector{Float64}:
 1.0
 0.0
 1.0
 1.0
 0.0
 1.0
 0.0
 0.0
 1.0
 0.0
 0.0
 0.0
 0.0

julia> round(sum(res.imputed.em); digits = 6)
5.0
```

!!! warning
    This code was generated by an LLM.
    Sam Abbott has only checked the interactive page, not this package.
    Treat the results as exploratory.
"""
function impute_linelist(
        onset::AbstractVector, report::AbstractVector;
        cutoff = nothing, delay = nothing, max_delay::Integer = DMAX,
        n_iter::Integer = 150, delay_period = nothing,
        delay_shrinkage::Real = 10.0
    )
    length(onset) == length(report) || throw(
        ArgumentError(
            "onset and report must have the same length, got " *
                "$(length(onset)) and $(length(report))"
        )
    )
    isempty(report) && throw(ArgumentError("the line list is empty"))
    day0 = nothing
    if any(x -> x isa Date, report) || any(x -> x isa Date, onset)
        onset, report, cutoff, day0 = _dates_to_days(onset, report, cutoff)
    end
    n_bad = count(
        x -> _is_missing_day(x) || (x isa Real && !isfinite(x)),
        report
    )
    n_bad > 0 && throw(
        ArgumentError(
            "report day is missing or infinite for $n_bad case(s); " *
                "every case needs a report day"
        )
    )
    all(x -> x isa Real, report) ||
        throw(ArgumentError("report days must be numbers or dates"))
    known = [x for x in onset if !_is_missing_day(x)]
    all(x -> x isa Real, known) ||
        throw(ArgumentError("onset days must be numbers, dates or missing"))
    any(isinf, known) && throw(
        ArgumentError(
            "onset days must not be infinite; use missing if unknown"
        )
    )
    all(_is_whole, report) ||
        throw(ArgumentError("report days must be whole numbers of days"))
    all(_is_whole, known) ||
        throw(ArgumentError("onset days must be whole numbers of days"))
    report_days = Int[Int(x) for x in report]
    if cutoff === nothing
        cutoff = maximum(report_days)
    elseif !(cutoff isa Real && _is_whole(cutoff))
        throw(ArgumentError("cutoff must be a whole number, got $cutoff"))
    end
    T = Int(cutoff)
    T >= 0 || throw(ArgumentError("cutoff must be non-negative, got $T"))
    max_delay >= 0 ||
        throw(ArgumentError("max_delay must be non-negative, got $max_delay"))
    n_iter >= 1 ||
        throw(ArgumentError("n_iter must be at least 1, got $n_iter"))
    delay !== nothing && delay_period !== nothing &&
        throw(
        ArgumentError(
            "give either delay or delay_period, not both"
        )
    )
    isfinite(delay_shrinkage) && delay_shrinkage > 0 ||
        throw(
        ArgumentError(
            "delay_shrinkage must be a positive finite " *
                "number, got $delay_shrinkage"
        )
    )
    fixed = delay === nothing ? nothing : _fixed_delay(delay, T, max_delay)
    is_matrix = delay isa AbstractMatrix
    fwd_pmf = is_matrix ? nothing : _check_delay(delay)
    period = is_matrix ? collect(0:T) : delay_periods(T, delay_period)

    n_days = T + 1
    complete = zeros(n_days)
    missing_by_report = zeros(n_days)
    empirical = zeros(max_delay + 1)
    onset_c, delay_c, report_m = Int[], Int[], Int[]
    dropped = 0
    for (o, t) in zip(onset, report_days)
        if !(0 <= t <= T)
            dropped += 1
        elseif _is_missing_day(o)
            push!(report_m, t)
            missing_by_report[t + 1] += 1
        else
            s = Int(o)
            d = t - s
            if s < 0 || d < 0 || d > max_delay
                dropped += 1
            else
                push!(onset_c, s)
                push!(delay_c, d)
                complete[s + 1] += 1
                empirical[d + 1] += 1
            end
        end
    end
    isempty(onset_c) && throw(
        ArgumentError(
            "no complete cases (cases with both an onset and a report day " *
                "in range) are left, so the delay cannot be estimated"
        )
    )
    empirical ./= sum(empirical)

    backsampling = backsampling_allocate(missing_by_report, empirical)
    shift = round_half_up(sum((0:max_delay) .* empirical))
    mean_alloc = mean_shift_allocate(missing_by_report, shift)
    fit = em_fit(
        onset_c, delay_c, report_m, T; max_delay, n_iter, fixed,
        period, delay_shrinkage
    )
    n_missing = sum(missing_by_report)
    lost(a) = max(n_missing - sum(a), 0.0)
    imputed = (
        backsampling = complete .+ backsampling,
        mean = complete .+ mean_alloc, em = complete .+ fit.allocation,
    )
    before_day0 = (
        backsampling = lost(backsampling),
        mean = n_missing - sum(mean_alloc),
    )
    if delay !== nothing
        fwd = is_matrix ?
            posterior_allocate(missing_by_report, ones(n_days), fixed) :
            backsampling_allocate(missing_by_report, fwd_pmf)
        imputed = merge(imputed, (fwd = complete .+ fwd,))
        before_day0 = merge(before_day0, (fwd = lost(fwd),))
    end
    return (
        complete = complete, missing_by_report = missing_by_report,
        empirical_delay = empirical, imputed = imputed,
        em_delay = fit.delay, em_delay_by_period = fit.delay_by_period,
        delay_period_start = fit.period_start,
        imputed_before_day0 = before_day0,
        dropped = dropped, cutoff = T, day0 = day0,
    )
end
