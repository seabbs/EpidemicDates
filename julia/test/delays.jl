@testitem "delay_pmf is a pmf on 0:dmax" begin
    for (μ, σ) in [(8, 8), (6.6, 6.072), (14, 9.8), (1, 0.5)]
        pmf = delay_pmf(μ, σ)
        @test length(pmf) == 61
        @test all(>=(0), pmf)
        @test sum(pmf) ≈ 1
    end
    @test length(delay_pmf(8, 8; dmax = 10)) == 11
    @test delay_pmf(8, 8; dmax = 0) == [1.0]
end

@testitem "delay_pmf mean is close to the Gamma mean" begin
    # Uniform primary censoring adds about half a day on average and
    # daily censoring of the report removes about half a day.
    for μ in [4.0, 8.0, 12.0]
        pmf = delay_pmf(μ, 0.5μ)
        @test isapprox(sum((0:60) .* pmf), μ; atol = 0.05)
    end
end

@testitem "delay_pmf matches a numerical double censored integral" begin
    using SpecialFunctions: gamma_inc
    μ, σ = 5.0, 3.0
    k, θ = (μ / σ)^2, σ^2 / μ
    F(x) = x <= 0 ? 0.0 : first(gamma_inc(k, x / θ))
    # P(d) = ∫_0^1 F(d + 1 - u) - F(d - u) du, by the midpoint rule
    u = ((1:20_000) .- 0.5) ./ 20_000
    direct = [sum(F.(d + 1 .- u) .- F.(d .- u)) / length(u) for d in 0:60]
    @test isapprox(delay_pmf(μ, σ), direct ./ sum(direct); atol = 1e-7)
end

@testitem "delay_pmf validates its inputs" begin
    @test_throws ArgumentError delay_pmf(0, 1)
    @test_throws ArgumentError delay_pmf(-1, 1)
    @test_throws ArgumentError delay_pmf(1, 0)
    @test_throws ArgumentError delay_pmf(Inf, 1)
    @test_throws ArgumentError delay_pmf(1, NaN)
    @test_throws ArgumentError delay_pmf(1, 1; dmax = -1)
end
