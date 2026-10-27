@testitem "impute_linelist returns every component" begin
    onset = [0, 2, missing, 5, missing]
    report = [3, 6, 7, 9, 12]
    res = impute_linelist(onset, report)
    @test res.cutoff == 12
    @test res.day0 === nothing
    @test keys(res.imputed) == (:backsampling, :mean, :em)
    for v in res.imputed
        @test length(v) == 13
    end
    @test res.complete == [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    @test res.missing_by_report[[8, 13]] == [1, 1]
    @test sum(res.missing_by_report) == 2
    @test res.empirical_delay[[4, 5]] == [1 / 3, 2 / 3]
    @test length(res.em_delay) == 61
    @test sum(res.em_delay) ≈ 1
    @test res.dropped == 0
    # mean delay is 11 / 3, which rounds to 4
    @test res.imputed.mean ==
          res.complete .+
          [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    @test sum(res.imputed.em) ≈ 5
    @test sum(res.imputed.backsampling) ≈ 5
end

@testitem "back-sampling is the expected allocation" begin
    res = impute_linelist([0, 0, missing], [1, 3, 5])
    # empirical delay: 1 and 3 days with probability 1/2 each
    expected = copy(res.complete)
    expected[[3, 5]] .+= 0.5  # onset days 2 and 4
    @test res.imputed.backsampling ≈ expected
    @test res.imputed_before_day0.backsampling ≈ 0
end

@testitem "mean shift rounds halves up" begin
    # mean delay 2.5 days, which rounds to 3, not to even (2)
    res = impute_linelist([0, 0, missing], [2, 3, 5])
    @test res.imputed.mean[3] == 1
    # a missing case reported before the shift is placed before day 0
    res = impute_linelist([0, 0, missing], [2, 3, 1]; cutoff = 5)
    @test res.imputed_before_day0.mean == 1
    @test sum(res.imputed.mean) == 2
end

@testitem "cases outside the cut-off or delay range are dropped" begin
    onset = [0, 5, 1, 1, missing, missing, 0]
    report = [2, 4, 90, -1, 3, 20, 8]
    res = impute_linelist(onset, report; cutoff = 10, max_delay = 5)
    # an onset after report, two reports out of range, a missing onset
    # reported after the cut-off, and a delay over max_delay
    @test res.dropped == 5
    @test sum(res.complete) == 1
    @test sum(res.missing_by_report) == 1
    @test length(res.empirical_delay) == 6
    @test length(res.em_delay) == 6
    res = impute_linelist([0, 0, -2], [2, 4, 3])
    @test res.dropped == 1
end

@testitem "a supplied delay adds the fwd method" begin
    pmf = delay_pmf(4, 3)
    res = impute_linelist([0, 2, missing, 5, missing], [3, 6, 7, 9, 12];
        delay = pmf)
    @test keys(res.imputed) == (:backsampling, :mean, :em, :fwd)
    @test haskey(res.imputed_before_day0, :fwd)
    alloc = res.imputed.fwd .- res.complete
    @test sum(alloc) + res.imputed_before_day0.fwd ≈ 2
    # an unnormalised delay gives the same result
    res2 = impute_linelist([0, 2, missing, 5, missing], [3, 6, 7, 9, 12];
        delay = 3 .* pmf)
    @test res2.imputed.fwd ≈ res.imputed.fwd
end

@testitem "Date input matches day input" begin
    using Dates
    d0 = Date(2026, 5, 8)
    onset = [d0, d0 + Day(2), missing, d0 + Day(5), missing]
    report = d0 .+ Day.([3, 6, 7, 9, 12])
    res = impute_linelist(onset, report)
    ref = impute_linelist([0, 2, missing, 5, missing], [3, 6, 7, 9, 12])
    @test res.day0 == d0
    @test res.imputed.backsampling == ref.imputed.backsampling
    @test res.imputed.mean == ref.imputed.mean
    @test res.imputed.em ≈ ref.imputed.em
    res = impute_linelist(onset, report; cutoff = d0 + Day(10))
    @test res.cutoff == 10
    # day 0 is the earliest date, which may be an onset
    res = impute_linelist([d0 - Day(3), missing], [d0, d0 + Day(1)])
    @test res.day0 == d0 - Day(3)
    @test res.cutoff == 4
end

@testitem "NaN onsets count as missing" begin
    a = impute_linelist([0.0, 2.0, NaN, 5.0, NaN], [3, 6, 7, 9, 12])
    b = impute_linelist([0, 2, missing, 5, missing], [3, 6, 7, 9, 12])
    @test a.imputed.mean == b.imputed.mean
    @test a.missing_by_report == b.missing_by_report
end

@testitem "impute_linelist validates its inputs" begin
    using Dates
    ok_onset, ok_report = [0, 2, missing], [3, 6, 7]
    @test_throws ArgumentError impute_linelist([0, 1], [1])
    @test_throws ArgumentError impute_linelist(Int[], Int[])
    @test_throws ArgumentError impute_linelist([0, 1], [1, missing])
    @test_throws ArgumentError impute_linelist([0, 1], [1.0, NaN])
    @test_throws ArgumentError impute_linelist([0, 1], [1.0, Inf])
    @test_throws ArgumentError impute_linelist([0.0, Inf], [1, 2])
    @test_throws ArgumentError impute_linelist([0.5, 1], [1, 2])
    @test_throws ArgumentError impute_linelist([0, 1], [1.5, 2])
    @test_throws ArgumentError impute_linelist(ok_onset, ok_report;
        cutoff = -1)
    @test_throws ArgumentError impute_linelist(ok_onset, ok_report;
        cutoff = 2.5)
    @test_throws ArgumentError impute_linelist(ok_onset, ok_report;
        n_iter = 0)
    @test_throws ArgumentError impute_linelist(ok_onset, ok_report;
        max_delay = -1)
    @test_throws ArgumentError impute_linelist(ok_onset, ok_report;
        delay = [0.0, 0.0])
    @test_throws ArgumentError impute_linelist(ok_onset, ok_report;
        delay = [0.5, -0.5])
    @test_throws ArgumentError impute_linelist(ok_onset, ok_report;
        delay = Float64[])
    # no complete cases left
    @test_throws ArgumentError impute_linelist([missing, missing], [1, 2])
    @test_throws ArgumentError impute_linelist([5, missing], [1, 2])
    # mixed dates and days
    @test_throws ArgumentError impute_linelist([0, missing],
        [Date(2026, 1, 1), Date(2026, 1, 2)])
    @test_throws ArgumentError impute_linelist(["a", missing], [1, 2])
end

@testitem "em_fit validates its inputs" begin
    using EpidemicDates: em_fit
    @test_throws ArgumentError em_fit([0], [1], [2], -1)
    @test_throws ArgumentError em_fit([0], [1], [2], 3; n_iter = 0)
    @test_throws ArgumentError em_fit([0, 1], [1], [2], 3)
    @test_throws ArgumentError em_fit([4], [0], [2], 3)
    @test_throws ArgumentError em_fit([0], [61], [2], 70)
    @test_throws ArgumentError em_fit([2], [2], [2], 3)
    @test_throws ArgumentError em_fit([0], [1], [4], 3)
    fit = em_fit([0, 1], [1, 2], [3, 3], 3)
    @test sum(fit.allocation) ≈ 2
    @test sum(fit.delay) ≈ 1
end

@testitem "internal helpers" begin
    using EpidemicDates: round_half_up, mean_shift_allocate,
                         backsampling_allocate
    @test round_half_up(2.5) == 3
    @test round_half_up(-0.5) == 0
    @test round_half_up(1.49) == 1
    @test_throws ArgumentError round_half_up(NaN)
    @test mean_shift_allocate([1.0, 2.0, 3.0], 1) == [2.0, 3.0, 0.0]
    @test mean_shift_allocate([1.0, 2.0], 5) == [0.0, 0.0]
    @test_throws ArgumentError mean_shift_allocate([1.0], -1)
    @test backsampling_allocate([0.0, 0.0, 4.0], [0.5, 0.25, 0.25]) ==
          [1.0, 1.0, 2.0]
end
