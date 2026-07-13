# Solve both AC-OPF forms through NLPModels → Percival, using the vectorized
# ArrayDiff derivatives. Percival's TRON subproblem needs Hessian-vector
# products; ArrayDiff is first-order, so we wrap the AL subproblem with an
# LBFGS quasi-Newton model (`subproblem_modifier = LBFGSModel`) which builds
# its Hessian approximation from gradient differences only.

include("build_percival.jl")
include(joinpath(ACOPF_DIR, "reference.jl"))

import NLPModelsModifiers
import Percival
import Printf

function solve_percival(nlp; kwargs...)
    return Percival.percival(
        nlp;
        subproblem_modifier = NLPModelsModifiers.LBFGSModel,
        kwargs...,
    )
end

function run_rect(; verbose = 1, kwargs...)
    d = case9mod()
    nlp, offset = build_rect_nlp(d)
    stats = solve_percival(nlp; verbose = verbose, kwargs...)
    return nlp, stats, offset
end

function run_polar(case = "case9.m"; verbose = 1, kwargs...)
    d = parse_polar_case(matpower_case(case))
    nlp, offset = build_polar_nlp(d)
    stats = solve_percival(nlp; verbose = verbose, kwargs...)
    return nlp, stats, offset
end

function report(name, stats, offset, ref)
    obj = stats.objective + offset
    Printf.@printf(
        "%-14s Percival obj %.2f (Ipopt %.2f, gap %.3f%%)  feas %.2e  status %s  iters %d\n",
        name,
        obj,
        ref,
        100 * (obj - ref) / abs(ref),
        stats.primal_feas,
        stats.status,
        stats.iter,
    )
    return obj
end
