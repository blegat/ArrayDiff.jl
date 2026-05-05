import SolverCore
import NLPModels
import Optimisers

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
