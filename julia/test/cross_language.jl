@testmodule Fixtures begin
    using JSON
    using EpidemicDates
    const PATH = joinpath(pkgdir(EpidemicDates), "..", "tests", "fixtures",
        "cross-language.json")
    const DATA = JSON.parsefile(PATH)
end

@testitem "delay pmfs match the shared fixtures" setup=[Fixtures] begin
    cases = Fixtures.DATA["delay_pmfs"]
    @test length(cases) == 3
    for case in cases
        ref = Float64.(case["pmf"])
        pmf = delay_pmf(case["mean"], case["mean"] * case["cv"];
            dmax = case["dmax"])
        @test length(pmf) == length(ref)
        @test isapprox(pmf, ref; atol = 1e-12, rtol = 0)
    end
end

@testitem "line list matches the shared fixtures" setup=[Fixtures] begin
    ll = Fixtures.DATA["linelist"]
    onset = [x === nothing ? missing : Int(x) for x in ll["onset"]]
    report = Int.(ll["report"])
    res = impute_linelist(onset, report; cutoff = ll["cutoff"],
        max_delay = ll["max_delay"])
    ref = ll["imputed"]
    @test isapprox(res.empirical_delay, Float64.(ll["empirical_delay"]);
        atol = 1e-14, rtol = 0)
    # back-sampling and mean must match exactly, up to rounding
    @test isapprox(res.imputed.backsampling,
        Float64.(ref["backsampling"]); atol = 1e-10, rtol = 0)
    @test res.imputed.mean == Float64.(ref["mean"])
    # em within 2% on weeks with at least 50 cases
    em = weekly(res.imputed.em)
    em_ref = weekly(Float64.(ref["em"]))
    big = em_ref .>= 50
    @test any(big)
    @test all(abs.(em[big] .- em_ref[big]) ./ em_ref[big] .<= 0.02)
end

@testitem "per-period EM matches the shared fixtures" setup=[Fixtures] begin
    ll = Fixtures.DATA["linelist_periods"]
    onset = [x === nothing ? missing : Int(x) for x in ll["onset"]]
    res = impute_linelist(onset, Int.(ll["report"]); cutoff = ll["cutoff"],
        max_delay = ll["max_delay"], delay_period = ll["delay_period"],
        delay_shrinkage = ll["delay_shrinkage"])
    ref = ll["imputed"]
    @test res.delay_period_start == Int.(ll["delay_period_start"])
    @test isapprox(res.imputed.backsampling,
        Float64.(ref["backsampling"]); atol = 1e-10, rtol = 0)
    @test res.imputed.mean == Float64.(ref["mean"])
    em = weekly(res.imputed.em)
    em_ref = weekly(Float64.(ref["em"]))
    big = em_ref .>= 50
    @test any(big)
    @test all(abs.(em[big] .- em_ref[big]) ./ em_ref[big] .<= 0.02)
end

@testitem "a supplied delay matrix matches the shared fixtures" setup=[Fixtures] begin
    ll = Fixtures.DATA["linelist_delay_matrix"]
    onset = [x === nothing ? missing : Int(x) for x in ll["onset"]]
    delay = permutedims(reduce(hcat, [Float64.(r) for r in ll["delay"]]))
    res = impute_linelist(onset, Int.(ll["report"]); cutoff = ll["cutoff"],
        max_delay = ll["max_delay"], delay)
    ref = ll["imputed"]
    @test isapprox(res.imputed.backsampling,
        Float64.(ref["backsampling"]); atol = 1e-10, rtol = 0)
    @test res.imputed.mean == Float64.(ref["mean"])
    @test isapprox(res.imputed.fwd, Float64.(ref["forward"]);
        atol = 1e-10, rtol = 0)
    em = weekly(res.imputed.em)
    em_ref = weekly(Float64.(ref["em"]))
    big = em_ref .>= 50
    @test any(big)
    @test all(abs.(em[big] .- em_ref[big]) ./ em_ref[big] .<= 0.02)
end
