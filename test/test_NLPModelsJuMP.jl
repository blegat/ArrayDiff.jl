module TestWithNLPModelsJuMP

using Test

using JuMP
using ArrayDiff
import MathOptInterface as MOI
import NLPModelsJuMP
import JSOSolvers

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

function _test_neural_nlpmodels_jump(solver)
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    model = Model(NLPModelsJuMP.Optimizer)
    set_attribute(model, "solver", solver)
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        ArrayDiff.Mode(),
    )
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
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) < 1e-6
    return
end

function test_neural_lbfgs()
    return _test_neural_nlpmodels_jump(JSOSolvers.LBFGSSolver)
end

function test_neural_trunkls()
    return _test_neural_nlpmodels_jump(JSOSolvers.TrunkSolverNLS)
end

function test_neural_tronls()
    return _test_neural_nlpmodels_jump(JSOSolvers.TronSolverNLS)
end

end

TestWithNLPModelsJuMP.runtests()
