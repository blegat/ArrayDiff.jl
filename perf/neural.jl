# Neural network optimization using ArrayDiff + NLopt
#
# This demonstrates end-to-end optimization of a simple two-layer neural
# network with array-valued decision variables, array-aware AD, and a
# first-order NLP solver.

using JuMP
using ArrayDiff
import Random
import NLopt
import NLPModelsJuMP

function bench(solver, ::Type{T} = Float64; h::Int = 4096, d::Int = 13, n::Int = 178, out_dim = 2, gpu::Bool = false) where {T<:Real}
    Random.seed!(0)
    X = randn(T, d, n)
    Y = randn(T, out_dim, n)

    model = GenericModel{T}(solver)
    V = gpu ? CUDA.CuVector{T} : Vector{T}
    set_attribute(model, MOI.AutomaticDifferentiationBackend(), ArrayDiff.Mode{V}())

    @variable(model, W1[1:h, 1:d], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:out_dim, 1:h], container = ArrayDiff.ArrayOfVariables)

    Y_hat = W2 * tanh.(W1 * X)
    loss = sum((Y_hat .- Y) .^ 2)
    @objective(model, Min, loss)

    for i in 1:n, j in 1:n
        set_start_value(W1[i, j], 0.1 * randn())
        set_start_value(W2[i, j], 0.1 * randn())
    end
    optimize!(model)

    if !is_solved_and_feasible(model)
        @warn(solution_summary(model))
    end
    return solve_time(model)
end

nlopt = optimizer_with_attributes(
    NLopt.Optimizer,
    "algorithm" => :LD_LBFGS,
    MOI.AutomaticDifferentiationBackend() => ArrayDiff.Mode(),
)

bench(nlopt)
