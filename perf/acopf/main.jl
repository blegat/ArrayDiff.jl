# CPU end-to-end validation of both vectorized AC-OPF forms.
#
#   julia --project=. main.jl
#
# Solves form 1 (rectangular, case9mod) and form 2 (polar, case9) with the
# first-order AL solver and compares objectives against Ipopt references.

include("ACOPF.jl")
include("reference.jl")

import Printf

function run_rect(; matrix = ELLMatrix, kwargs...)
    d = case9mod()
    prob = build_rect(d; matrix)
    x, stats = solve!(prob; kwargs...)
    return prob, x, stats
end

function run_polar(case::AbstractString = "case9.m"; use_gather = true, kwargs...)
    d = parse_polar_case(matpower_case(case))
    prob = build_polar(d; use_gather)
    x, stats = solve!(prob; kwargs...)
    return d, prob, x, stats
end

function run_all()
    println("═"^70)
    println("Form 1: rectangular voltages + Ybus (JuMP tutorial, case9mod)")
    println("═"^70)
    ref1 = rect_reference(case9mod())
    Printf.@printf("Ipopt reference objective: %.2f\n", ref1)
    _, _, stats1 = run_rect()
    Printf.@printf(
        "first-order AL objective: %.2f  (gap %.3f%%, viol %.2e)\n",
        stats1.obj,
        100 * (stats1.obj - ref1) / abs(ref1),
        stats1.viol,
    )
    println()
    println("═"^70)
    println("Form 2: polar voltages, sin/cos branch flows (GenOpt, case9)")
    println("═"^70)
    ref2 = polar_reference(matpower_case("case9.m"))
    Printf.@printf("Ipopt/PowerModels reference objective: %.2f\n", ref2)
    _, _, _, stats2 = run_polar("case9.m")
    Printf.@printf(
        "first-order AL objective: %.2f  (gap %.3f%%, viol %.2e)\n",
        stats2.obj,
        100 * (stats2.obj - ref2) / abs(ref2),
        stats2.viol,
    )
    return (ref1, stats1, ref2, stats2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
