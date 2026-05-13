include("ReverseAD.jl")
include("ArrayDiff.jl")
include("JuMP.jl")
if VERSION >= v"1.11"
    # [sources] not supported on Julia v1.10
    # Needs https://github.com/jump-dev/NLopt.jl/pull/273
    include("NLopt.jl")
    # Needs https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl/pull/229
    include("NLPModelsJuMP.jl")
    include("Optimisers.jl")
    include("Optimisers_GPU.jl")
end
