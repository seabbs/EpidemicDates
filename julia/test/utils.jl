@testitem "weekly sums days into weeks" begin
    @test weekly(ones(14)) == [7.0, 7.0]
    @test weekly(1:10) == [28.0, 27.0]
    @test weekly(Float64[]) == Float64[]
    x = rand(23)
    @test sum(weekly(x)) ≈ sum(x)
    @test length(weekly(x)) == 4
end

@testitem "correct_truncation divides by the fraction reported" begin
    pmf = [0.2, 0.3, 0.5]
    counts = fill(10.0, 5)
    out = correct_truncation(counts, pmf, 4)
    @test out ≈ [10.0, 10.0, 10.0, 10 / 0.5, 10 / 0.2]
    # days after the cut-off and days below min_cover are set to 0
    out = correct_truncation(counts, pmf, 3; min_cover = 0.25)
    @test out ≈ [10.0, 10.0, 10 / 0.5, 0.0, 0.0]
    # unnormalised delays are normalised
    @test correct_truncation(counts, 2 .* pmf, 4) ≈
          correct_truncation(counts, pmf, 4)
end

@testitem "correct_truncation recovers truncated expected counts" begin
    pmf = delay_pmf(6, 4)
    T = 40
    truth = 100 .* exp.(0.05 .* (0:T))
    cdf = cumsum(pmf)
    observed = [truth[s + 1] * cdf[min(T - s, 60) + 1] for s in 0:T]
    out = correct_truncation(observed, pmf, T)
    @test out[1:(end - 1)] ≈ truth[1:(end - 1)]
end

@testitem "correct_truncation validates its inputs" begin
    @test_throws ArgumentError correct_truncation([1.0], [1.0], -1)
    @test_throws ArgumentError correct_truncation([1.0], [1.0], 0;
        min_cover = 0)
    @test_throws ArgumentError correct_truncation([1.0], [0.0], 0)
    @test_throws ArgumentError correct_truncation([1.0], [-1.0, 2.0], 0)
end
