# Compare NLPModels solvers on the AC-OPF `ArrayDiffNLPModel` (vectorized
# ArrayDiff derivatives), against Ipopt references. Everything goes through the
# NLPModels interface: obj/grad from the objective evaluator, cons/jprod/jtprod
# from the vectorized residual evaluators, and (for MadNLP) an explicit
# Jacobian materialized from jtprod.
#
#   * MadNLP  — interior point + CompactLBFGS quasi-Newton (no exact Hessian).
#   * Percival — augmented Lagrangian; LBFGSModel subproblems + TRON (CPU) or
#     the new SPGSubSolver (GPU-compatible).

include("build_percival.jl")
include(joinpath(ACOPF_DIR, "reference.jl"))

import MadNLP
import NLPModelsModifiers
import Percival
import Printf

function madnlp_solve(nlp; kwargs...)
    return MadNLP.madnlp(
        nlp;
        hessian_approximation = MadNLP.CompactLBFGS,
        print_level = MadNLP.ERROR,
        kwargs...,
    )
end

percival_lbfgs(nlp; kwargs...) = Percival.percival(
    nlp;
    subproblem_modifier = NLPModelsModifiers.LBFGSModel,
    verbose = 0,
    kwargs...,
)

function line(name, obj, ref, feas, status, iters)
    Printf.@printf(
        "  %-18s obj %12.2f  gap %8.4f%%  feas %9.2e  %-16s it %d\n",
        name, obj, 100 * (obj - ref) / abs(ref), feas, status, iters,
    )
    return
end

function compare_rect()
    d = case9mod()
    ref = rect_reference(d)
    println("── rect (case9mod), Ipopt ref $(round(ref; digits=2))")
    nlp, off = build_rect_nlp(d)
    s = madnlp_solve(nlp; max_iter = 1000)
    line("MadNLP/LBFGS", s.objective + off, ref, s.primal_feas, s.status, s.iter)
    nlp, off = build_rect_nlp(d)
    s = percival_lbfgs(nlp; max_iter = 300, max_time = 60.0)
    line("Percival/LBFGS+Tron", s.objective + off, ref, s.primal_feas, s.status, s.iter)
    return
end

function compare_polar(case = "case9.m")
    d = parse_polar_case(matpower_case(case))
    ref = polar_reference(matpower_case(case))
    println("── polar ($case), Ipopt ref $(round(ref; digits=2))")
    nlp, off = build_polar_nlp(d)
    s = madnlp_solve(nlp; max_iter = 1000)
    line("MadNLP/LBFGS", s.objective + off, ref, s.primal_feas, s.status, s.iter)
    nlp, off = build_polar_nlp(d)
    s = percival_lbfgs(nlp; max_iter = 300, max_time = 60.0)
    line("Percival/LBFGS+Tron", s.objective + off, ref, s.primal_feas, s.status, s.iter)
    return
end

function compare_all()
    compare_rect()
    compare_polar("case9.m")
    compare_polar("case14.m")
    compare_polar("case30.m")
    return
end
