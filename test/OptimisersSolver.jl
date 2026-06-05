import LinearAlgebra
import SolverCore
import NLPModels
import Optimisers

# An NLPModels solver that runs an `Optimisers.AbstractRule` (e.g. `Adam`) on
# the variable vector of an unconstrained `AbstractNLPModel` using `obj` and
# `grad!`. Designed to be plugged into `NLPModelsJuMP.Optimizer` via
# `set_attribute(model, "solver", OptimisersSolver)`.
mutable struct OptimisersSolver{R<:Optimisers.AbstractRule,V<:AbstractVector} <:
               SolverCore.AbstractOptimizationSolver
    rule::R
    x::V
    g::V
end

function OptimisersSolver(
    nlp::NLPModels.AbstractNLPModel{T,V};
    rule::Optimisers.AbstractRule = Optimisers.Adam(T(0.05)),
) where {T,V<:AbstractVector{T}}
    nvar = NLPModels.get_nvar(nlp.meta)
    x = similar(NLPModels.get_x0(nlp.meta), nvar)
    g = similar(x)
    fill!(x, zero(T))
    fill!(g, zero(T))
    return OptimisersSolver(rule, x, g)
end

function SolverCore.reset!(solver::OptimisersSolver)
    fill!(solver.x, zero(eltype(solver.x)))
    fill!(solver.g, zero(eltype(solver.g)))
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
