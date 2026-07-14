module TestWithNLPModelsJuMP

using Test

using JuMP
using ArrayDiff
import MathOptInterface as MOI
import NLPModelsJuMP
import JSOSolvers
import NLPModelsModifiers
import Percival

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

# Constrained solve through the full pipeline: `@constraint(model, expr in
# MOI.Zeros(m))` over a vectorized array expression → NLPModelsJuMP collects it
# into a constrained `ArrayDiffNLPModel` (one residual evaluator per vector
# constraint) → Percival (augmented Lagrangian; LBFGS subproblems since
# ArrayDiff is first-order).
#
# Projection problem with an analytic solution:
#   min ‖x − t‖²  s.t.  A x = b   ⇒   x* = t − A' (A A')⁻¹ (A t − b)
function test_vector_constraint_solve()
    if !isdefined(NLPModelsJuMP, :_try_array_nlp_model)
        @info "NLPModelsJuMP has no `_try_array_nlp_model`; skipping the " *
              "constrained test (needs the updated bl/arraydiff branch)."
        return
    end
    model = Model(NLPModelsJuMP.Optimizer)
    set_attribute(
        model,
        "solver",
        nlp -> Percival.PercivalSolver(
            nlp;
            subproblem_modifier = NLPModelsModifiers.LBFGSModel,
        ),
    )
    set_attribute(model, "subproblem_modifier", NLPModelsModifiers.LBFGSModel)
    set_attribute(model, MOI.AutomaticDifferentiationBackend(), ArrayDiff.Mode())
    set_silent(model)
    @variable(model, x[1:3], container = ArrayDiff.ArrayOfVariables)
    A = [1.0 1.0 1.0; 1.0 -1.0 0.0]
    b = [1.0, 0.0]
    t = [2.0, 1.0, 0.5]
    @objective(model, Min, sum((x .- t) .^ 2))
    @constraint(model, A * x .- b in MOI.Zeros(2))
    optimize!(model)
    xstar = t - A' * ((A * A') \ (A * t - b))
    @test isapprox(value.(x), xstar; atol = 1e-4)
    @test isapprox(objective_value(model), sum((xstar .- t) .^ 2); atol = 1e-4)
    return
end

end

TestWithNLPModelsJuMP.runtests()
