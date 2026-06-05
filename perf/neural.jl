# Neural network optimization using ArrayDiff + NLopt #
# This demonstrates end-to-end optimization of a simple two-layer neural
# network with array-valued decision variables, array-aware AD, and a
# first-order NLP solver.

using JuMP
using ArrayDiff
import Random

# Benchmark used for SIAM'OP 26 talk.
function bench(solver, ::Type{T} = Float64; h::Int = 4096, d::Int = 13, n::Int = 178, out_dim = 2, gpu::Bool = false) where {T<:Real}
    Random.seed!(0)
    X = randn(T, d, n)
    Y = randn(T, out_dim, n)

    model = GenericModel{T}(solver)
    @variable(model, W1[1:h, 1:d],
        container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:out_dim, 1:h],
        container = ArrayDiff.ArrayOfVariables)
    Y_hat = W2 * tanh.(W1 * X)
    # We need `.-` and not `-` as a workaround for
    # https://github.com/blegat/ArrayDiff.jl/issues/83
    loss = sum((Y_hat .- Y) .^ 2)
    @objective(model, Min, loss)

    for i in 1:n, j in 1:n
        set_start_value(W1[i, j], 0.1 * randn())
        set_start_value(W2[i, j], 0.1 * randn())
    end
    V = gpu ? CUDA.CuVector{T} : Vector{T}
    set_attribute(model,
        MOI.AutomaticDifferentiationBackend(),
        ArrayDiff.Mode{V}())
    optimize!(model)

    display(solution_summary(model))
    if !is_solved_and_feasible(model)
        @warn(solution_summary(model))
    end
    return model
    return solve_time(model)
end

import NLopt
nlopt = optimizer_with_attributes(
    NLopt.Optimizer,
    "algorithm" => :LD_LBFGS,
    "ftol_rel" => 1e-14,                                                      
    "ftol_abs" => 1e-14,
    "xtol_rel" => 1e-14,                                                      
    "maxeval"  => 100_000,
)
m = bench(nlopt)

import NLPModelsJuMP
include(joinpath(dirname(dirname(pathof(ArrayDiff))), "test", "OptimisersSolver.jl"))

import JSOSolvers
lbfgs = optimizer_with_attributes(
    NLPModelsJuMP.Optimizer,
    "solver" => JSOSolvers.lbfgs,
)
bench(lbfgs)

adam = optimizer_with_attributes(
    NLPModelsJuMP.Optimizer,
    "tol" => 1e-4,
    "solver" => OptimisersSolver,
)
bench(adam)

import CUDA
bench(adam, Float32, gpu = true)
