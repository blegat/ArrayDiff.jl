import ArrayDiff
import ParallelTestRunner

is_test_file(f) = startswith(f, "test_") && endswith(f, ".jl")

testsuite = Dict{String,Expr}()
for file in filter(is_test_file, readdir(@__DIR__))
    name = file[1:end-3]
    if VERSION < v"1.11" && name in (
        # [sources] not supported on Julia v1.10
        # Needs https://github.com/jump-dev/NLopt.jl/pull/273
        "test_NLopt",
        # Needs https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl/pull/229
        "test_NLPModelsJuMP",
        "test_Optimisers",
    )
        continue
    end
    testsuite[name] = :(include($(joinpath(@__DIR__, file))))
end

ParallelTestRunner.runtests(ArrayDiff, ARGS; testsuite)
