module TestWithOptimisers

using Test

using JuMP
using ArrayDiff
import LinearAlgebra
import MathOptInterface as MOI
import NLPModelsJuMP

include(joinpath(@__DIR__, "OptimisersSolver.jl"))

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_neural_optimisers()
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    model = Model(NLPModelsJuMP.Optimizer)
    set_attribute(model, "solver", OptimisersSolver)
    set_attribute(model, MOI.AutomaticDifferentiationBackend(), ArrayDiff.Mode())
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    # Use distinct starting values to break symmetry
    start_W1 = [0.3 -0.2; 0.1 0.4]
    start_W2 = [-0.1 0.5; 0.2 -0.3]
    for i in 1:n, j in 1:n
        set_start_value(W1[i, j], start_W1[i, j])
        set_start_value(W2[i, j], start_W2[i, j])
    end
    Y = W2 * tanh.(W1 * X)
    loss = sum((Y .- target) .^ 2)
    @objective(model, Min, loss)
    set_attribute(model, "max_iter", 20_000)
    set_attribute(model, "tol", 1e-6)
    optimize!(model)
    @test objective_value(model) < 1e-3
    return
end

end

TestWithOptimisers.runtests()
