# Solve the AC-OPF ArrayDiffNLPModel with MadNLP in quasi-Newton mode
# (`hessian_approximation = CompactLBFGS`), so no exact Hessian is needed —
# ArrayDiff supplies the objective gradient and the constraint Jacobian
# (materialized via `jac_coord!`). MadNLP is an interior-point method (a
# different family than Percival's augmented Lagrangian), and on CPU it
# converges tightly. The same model type also feeds MadNLPGPU on real CUDA
# hardware (untested here — no GPU in this container).

include("build_percival.jl")
include(joinpath(ACOPF_DIR, "reference.jl"))

import MadNLP
import Printf

function solve_madnlp(nlp; quasi_newton = true, print_level = MadNLP.INFO, kwargs...)
    opts = Dict{Symbol, Any}(kwargs)
    if quasi_newton
        opts[:hessian_approximation] = MadNLP.CompactLBFGS
    end
    return MadNLP.madnlp(nlp; print_level = print_level, opts...)
end

function madnlp_rect(; kwargs...)
    d = case9mod()
    nlp, off = build_rect_nlp(d)
    stats = solve_madnlp(nlp; kwargs...)
    return nlp, stats, off
end

function madnlp_polar(case = "case9.m"; kwargs...)
    d = parse_polar_case(matpower_case(case))
    nlp, off = build_polar_nlp(d)
    stats = solve_madnlp(nlp; kwargs...)
    return nlp, stats, off
end

function report_madnlp(name, stats, off, ref)
    obj = stats.objective + off
    Printf.@printf(
        "%-16s MadNLP obj %.2f (Ipopt %.2f, gap %.4f%%)  status %s  iters %d\n",
        name,
        obj,
        ref,
        100 * (obj - ref) / abs(ref),
        stats.status,
        stats.iter,
    )
    return obj
end
