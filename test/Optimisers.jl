module TestWithOptimisers

using Test

using JuMP
using ArrayDiff
import LinearAlgebra
import MathOptInterface as MOI
import NLPModels
import NLPModelsJuMP
import Optimisers
import SolverCore

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

# An NLPModels solver that runs an `Optimisers.AbstractRule` (e.g. `Adam`) on
# the variable vector of an unconstrained `AbstractNLPModel` using `obj` and
# `grad!`. Designed to be plugged into `NLPModelsJuMP.Optimizer` via
# `set_attribute(model, "solver", OptimisersSolver)`.
mutable struct OptimisersSolver{R<:Optimisers.AbstractRule} <:
               SolverCore.AbstractOptimizationSolver
    rule::R
    x::Vector{Float64}
    g::Vector{Float64}
end

function OptimisersSolver(
    nlp::NLPModels.AbstractNLPModel;
    rule::Optimisers.AbstractRule = Optimisers.Adam(0.05),
)
    nvar = NLPModels.get_nvar(nlp.meta)
    return OptimisersSolver(rule, zeros(Float64, nvar), zeros(Float64, nvar))
end

function SolverCore.reset!(solver::OptimisersSolver)
    fill!(solver.x, 0.0)
    fill!(solver.g, 0.0)
    return solver
end

function SolverCore.reset!(
    solver::OptimisersSolver,
    nlp::NLPModels.AbstractNLPModel,
)
    return SolverCore.reset!(solver)
end

function SolverCore.solve!(
    solver::OptimisersSolver,
    nlp::NLPModels.AbstractNLPModel,
    stats::SolverCore.GenericExecutionStats;
    max_iter::Int = 10_000,
    tol::Real = 1e-6,
    verbose::Int = 0,
)
    SolverCore.reset!(stats)
    copyto!(solver.x, NLPModels.get_x0(nlp.meta))
    state = Optimisers.setup(solver.rule, solver.x)
    start = time()
    iter = 0
    status = :max_iter
    while iter < max_iter
        NLPModels.grad!(nlp, solver.x, solver.g)
        if LinearAlgebra.norm(solver.g) < tol
            status = :first_order
            break
        end
        state, solver.x = Optimisers.update!(state, solver.x, solver.g)
        iter += 1
        if verbose > 0 && iter % verbose == 0
            @info "Optimisers" iter obj = NLPModels.obj(nlp, solver.x)
        end
    end
    SolverCore.set_iter!(stats, iter)
    SolverCore.set_status!(stats, status)
    SolverCore.set_solution!(stats, solver.x)
    SolverCore.set_objective!(stats, NLPModels.obj(nlp, solver.x))
    SolverCore.set_time!(stats, time() - start)
    return stats
end

function _test_neural_optimisers(with_norm::Bool)
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
    if with_norm
        loss = LinearAlgebra.norm(Y .- target)
    else
        loss = sum((Y .- target) .^ 2)
    end
    @objective(model, Min, loss)
    set_attribute(model, "max_iter", 20_000)
    set_attribute(model, "tol", 1e-6)
    optimize!(model)
    @test objective_value(model) < 1e-3
    return
end

function test_neural_optimisers()
    _test_neural_optimisers(true)
    return _test_neural_optimisers(false)
end

end

TestWithOptimisers.runtests()
