# A complete polar AC-OPF written as a JuMP model, using ArrayDiff's vectorized
# array variables and the new vector `@constraint(..., expr in set)` support.
# Every power-flow equation is ONE vectorized constraint (an
# `ArrayNonlinearFunction in MOI.Zeros`), not `nbranch` scalar rows.
#
#   julia --project=perf/percival
#   include("perf/percival/jump_acopf.jl")
#   model = build_jump_polar("case9.m")          # PowerModels-bundled, offline
#   # inspect what was added:
#   describe_constraints(model)
#
# Status: this BUILDS the JuMP/MOI model — variables, a scalar objective, and
# the vectorized constraints are all stored on the backend. Solving it end to
# end (JuMP → NLPModelsJuMP → an NLPModels solver) still needs the
# NLPModelsJuMP step that collects these `ArrayNonlinearFunction in Zeros`
# constraints into an `ArrayDiffNLPModel`; until then use `build_polar_nlp`
# (build_percival.jl), which assembles the same model directly.

include("build_percival.jl")   # parse_polar_case, structured matrices, matpower_case

import JuMP
import MathOptInterface as MOI

_case(name) = isfile(matpower_case(name)) ? matpower_case(name) : name

function build_jump_polar(name::AbstractString)
    d = parse_polar_case(_case(name))
    nb, ng, nl = d.nbus, d.ngen, d.nbranch
    Fg = GatherMatrix(d.f_bus, nb)
    Tg = GatherMatrix(d.t_bus, nb)
    Cg = GatherMatrix(d.gen_bus, nb)
    Ft, Tt, Ct = transpose(Fg), transpose(Tg), transpose(Cg)

    model = JuMP.Model()
    JuMP.@variable(model, va[1:nb], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, vm[1:nb], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, pg[1:ng], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, qg[1:ng], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, pf[1:nl], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, pt[1:nl], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, qf[1:nl], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, qt[1:nl], container = ArrayDiff.ArrayOfVariables)

    # Bounds (ArrayOfVariables is indexable → set per block).
    _set_bounds(vm, d.vmin, d.vmax)
    _set_bounds(pg, d.pmin, d.pmax)
    _set_bounds(qg, d.qmin, d.qmax)
    _set_bounds(pf, -d.rate_a, d.rate_a)
    _set_bounds(pt, -d.rate_a, d.rate_a)
    _set_bounds(qf, -d.rate_a, d.rate_a)
    _set_bounds(qt, -d.rate_a, d.rate_a)
    for r in d.ref_buses         # reference-bus angle fixed to 0
        JuMP.fix(va[r], 0.0)
    end

    # Vectorized branch/bus quantities.
    vmf, vmt = Fg * vm, Tg * vm
    Δ = Fg * va .- Tg * va
    cΔ, sΔ = cos.(Δ), sin.(Δ)
    vv = vmf .* vmt

    # Objective: generation cost (scalar reduction of array expressions).
    JuMP.@objective(model, Min, sum(d.cost1 .* pg .^ 2 .+ d.cost2 .* pg))

    # Power-flow equalities — one vector constraint each.
    JuMP.@constraint(model, pf .- (d.c5 .* vmf .^ 2 .+ d.c3 .* (vv .* cΔ) .+ d.c4 .* (vv .* sΔ)) in MOI.Zeros(nl))
    JuMP.@constraint(model, qf .+ d.c6 .* vmf .^ 2 .+ d.c4 .* (vv .* cΔ) .- d.c3 .* (vv .* sΔ) in MOI.Zeros(nl))
    JuMP.@constraint(model, pt .- (d.c7 .* vmt .^ 2 .+ d.c1 .* (vv .* cΔ) .- d.c2 .* (vv .* sΔ)) in MOI.Zeros(nl))
    JuMP.@constraint(model, qt .+ d.c8 .* vmt .^ 2 .+ d.c2 .* (vv .* cΔ) .+ d.c1 .* (vv .* sΔ) in MOI.Zeros(nl))
    # Nodal power balance (scatter branch flows / generation to buses).
    JuMP.@constraint(model, Ft * pf .+ Tt * pt .- Ct * pg .+ d.pd .+ d.gs .* vm .^ 2 in MOI.Zeros(nb))
    JuMP.@constraint(model, Ft * qf .+ Tt * qt .- Ct * qg .+ d.qd .- d.bs .* vm .^ 2 in MOI.Zeros(nb))
    # Thermal limits |S|² ≤ rate² — one vector inequality each.
    JuMP.@constraint(model, pf .^ 2 .+ qf .^ 2 .- d.rate_a_sq in MOI.Nonpositives(nl))
    JuMP.@constraint(model, pt .^ 2 .+ qt .^ 2 .- d.rate_a_sq in MOI.Nonpositives(nl))
    return model
end

function _set_bounds(v, lo, hi)
    for i in eachindex(lo)
        JuMP.set_lower_bound(v[i], lo[i])
        JuMP.set_upper_bound(v[i], hi[i])
    end
    return
end

# Show the vectorized constraints stored on the model's backend.
function describe_constraints(model)
    b = JuMP.backend(model)
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        n = MOI.get(model, MOI.NumberOfConstraints{F,S}())
        println("  ", n, " × ", F, " in ", S)
    end
    return
end
