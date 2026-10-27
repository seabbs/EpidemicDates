@testmodule Falling begin
    using EpidemicDates
    # A line list of rounded expected counts with steady growth and a mean
    # delay falling linearly from 12 to 5 days, cut at `cutoff`. Every 5th
    # and 5th + 1 case misses its onset.
    function simulate(cutoff = 120)
        days = 0:cutoff
        delay = permutedims(
            reduce(
                hcat,
                [delay_pmf(m, 0.8 * m) for m in 12 .- 7 .* days ./ cutoff]
            )
        )
        onset = Union{Int, Missing}[]
        report = Int[]
        reported = zeros(cutoff + 1)
        eventual = zeros(cutoff + 1)
        i = 0
        for s in days, d in 0:60

            n = round(Int, 100 * exp(0.03 * s) * delay[s + 1, d + 1])
            eventual[s + 1] += n
            s + d > cutoff && continue
            reported[s + 1] += n
            for _ in 1:n
                push!(report, s + d)
                push!(onset, i % 5 < 2 ? missing : s)
                i += 1
            end
        end
        return (; onset, report, cutoff, delay, reported, eventual)
    end
    # Full weeks ending on the last day.
    function weeks(x)
        n = length(x) ÷ 7
        tail = x[(end - 7 * n + 1):end]
        return vec(sum(reshape(tail, 7, n); dims = 1))
    end
    # Per-day delay pmfs from per-period estimates.
    function by_day(res)
        period = [
            searchsortedlast(res.delay_period_start, s)
                for s in 0:res.cutoff
        ]
        return res.em_delay_by_period[period, :]
    end
end

@testitem "delay_periods counts periods back from the cut-off" begin
    @test EpidemicDates.delay_periods(9) == zeros(Int, 10)
    @test EpidemicDates.delay_periods(9, 4) == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    @test EpidemicDates.delay_periods(9, 10) == zeros(Int, 10)
    @test_throws ArgumentError EpidemicDates.delay_periods(9, 0)
    @test_throws ArgumentError EpidemicDates.delay_periods(9, 2.5)
end

@testitem "per-period EM beats static EM when delays fall" setup = [Falling] begin
    ll = Falling.simulate()
    weeks = Falling.weeks
    function errors(period)
        res = impute_linelist(
            ll.onset, ll.report; cutoff = ll.cutoff,
            delay_period = period
        )
        imp = sum(abs.(weeks(res.imputed.em) .- weeks(ll.reported))) /
            sum(ll.reported)
        nowcast = correct_truncation(
            res.imputed.em, Falling.by_day(res),
            ll.cutoff; min_cover = 1.0e-3
        )
        rel = (weeks(nowcast) ./ weeks(ll.eventual) .- 1)[(end - 3):end]
        return imp, sum(abs.(rel)) / 4, res
    end
    imp_s, now_s, _ = errors(nothing)
    imp_p, now_p, res = errors(14)
    @test imp_p < imp_s
    @test now_p < 0.75 * now_s
    means = res.em_delay_by_period * (0:60)
    @test size(res.em_delay_by_period) == (9, 61)
    @test means[1] > means[end] + 2
end

@testitem "the default is one static delay and one period matches it" setup = [Falling] begin
    ll = Falling.simulate(80)
    res = impute_linelist(ll.onset, ll.report; cutoff = 80)
    @test res.em_delay_by_period == permutedims(res.em_delay)
    @test res.delay_period_start == [0]
    one = impute_linelist(ll.onset, ll.report; cutoff = 80, delay_period = 81)
    @test one.imputed.em == res.imputed.em
    @test one.em_delay == res.em_delay
end

@testitem "a period with no cases takes the previous period's delay" begin
    onset = repeat(vcat(0:13, 28:41); inner = 5)
    delays = vcat(repeat([0, 1, 2, 3, 5], 14), repeat([2, 4, 6, 7, 9], 14))
    res = impute_linelist(
        onset, onset .+ delays; cutoff = 55,
        delay_period = 14
    )
    f = res.em_delay_by_period
    @test res.delay_period_start == [0, 14, 28, 42]
    @test isapprox(f[2, :], f[1, :]; atol = 1.0e-9)
    @test isapprox(f[4, :], f[3, :]; atol = 1.0e-9)
    @test maximum(abs.(f[3, :] .- f[1, :])) > 1.0e-3
end

@testitem "a short line list works with periods" begin
    res = impute_linelist(
        [0, 2, missing, 5, missing], [3, 6, 7, 9, 12];
        delay_period = 4
    )
    @test size(res.em_delay_by_period) == (4, 61)
    @test all(isapprox.(sum(res.em_delay_by_period; dims = 2), 1))
    @test sum(res.imputed.em) ≈ 5
end

@testitem "a given delay is held fixed by em" begin
    pmf = delay_pmf(4, 3)
    res = impute_linelist(
        [0, 2, missing, 5, missing], [3, 6, 7, 9, 12];
        delay = pmf
    )
    @test res.em_delay ≈ pmf
    @test_throws ArgumentError impute_linelist(
        [0, missing], [2, 3];
        delay = pmf, delay_period = 2
    )
    @test_throws ArgumentError impute_linelist(
        [0, missing], [2, 3];
        delay_shrinkage = 0
    )
end

@testitem "a delay matrix is a plug-in and beats a static plug-in" setup = [Falling] begin
    ll = Falling.simulate()
    weeks = Falling.weeks
    res = impute_linelist(
        ll.onset, ll.report; cutoff = ll.cutoff,
        delay = ll.delay
    )
    @test size(res.em_delay_by_period) == (121, 61)
    @test res.em_delay_by_period[8, :] ≈ ll.delay[8, :] ./ sum(ll.delay[8, :])
    # fwd renormalises over onset days 0:t, so every case is placed
    @test sum(res.imputed.fwd .- res.complete) ≈ sum(res.missing_by_report)
    function err(delay)
        fit = impute_linelist(ll.onset, ll.report; cutoff = ll.cutoff, delay)
        nowcast = correct_truncation(
            fit.imputed.em, delay, ll.cutoff;
            min_cover = 1.0e-3
        )
        rel = (weeks(nowcast) ./ weeks(ll.eventual) .- 1)[(end - 3):end]
        return sum(abs.(rel)) / 4
    end
    @test err(ll.delay) < 0.5 * err(delay_pmf(8, 6.4))
    @test_throws ArgumentError impute_linelist(
        ll.onset, ll.report;
        cutoff = ll.cutoff, delay = ll.delay[2:end, :]
    )
end

@testitem "correct_truncation takes a delay matrix" begin
    delay = [0.5 0.5; 1.0 0.0; 0.25 0.75]
    @test correct_truncation([4.0, 4.0, 4.0], delay, 2) == [4.0, 4.0, 16.0]
    @test_throws ArgumentError correct_truncation([4.0, 4.0], delay, 1)
end
