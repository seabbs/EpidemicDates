@testitem "Aqua quality checks" begin
    using Aqua
    Aqua.test_all(EpidemicDates)
end

@testitem "Code is formatted with JuliaFormatter" begin
    using JuliaFormatter
    @test JuliaFormatter.format(pkgdir(EpidemicDates); overwrite = false)
end

@testitem "Docstring examples run (doctests)" begin
    using Documenter
    DocMeta.setdocmeta!(EpidemicDates, :DocTestSetup,
        :(using EpidemicDates); recursive = true)
    doctest(EpidemicDates; manual = false)
end
