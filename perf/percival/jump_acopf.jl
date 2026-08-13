# A complete polar AC-OPF written as a JuMP model, using ArrayDiff's vectorized
# array variables and the new vector `@constraint(..., expr in set)` support.
# Every power-flow equation is ONE vectorized constraint (an
# `ArrayNonlinearFunction in MOI.Zeros`), not `nbranch` scalar rows.
#
#   julia --project=perf/percival
#   include("perf/percival/jump_acopf.jl")
#   import Percival, NLPModelsModifiers, NLPModelsJuMP
#   model = build_jump_polar("case9.m"; slacks = true)  # bundled case, offline
#   set_optimizer(model, NLPModelsJuMP.Optimizer)
#   set_attribute(model, "solver",
#       nlp -> Percival.PercivalSolver(nlp;
#           subproblem_modifier = NLPModelsModifiers.LBFGSModel))
#   set_attribute(model, "subproblem_modifier", NLPModelsModifiers.LBFGSModel)
#   set_attribute(model, MOI.AutomaticDifferentiationBackend(), ArrayDiff.Mode())
#   optimize!(model)                 # case9: 347.67 vs Ipopt 347.70 (−0.008%)
#   objective_value(model); value.(vm)
#
# NLPModelsJuMP (branch bl/arraydiff) collects the `ArrayNonlinearFunction in
# Zeros/Nonpositives` constraints into a constrained `ArrayDiffNLPModel` (one
# vectorized residual evaluator per constraint), which any NLPModels solver
# consumes. With Percival use `slacks = true` (its augmented Lagrangian wants
# equality constraints + bounds); `build_polar_nlp` (build_percival.jl) still
# assembles the same NLPModel directly without going through MOI.

include("build_percival.jl")   # parse_polar_case, structured matrices, matpower_case

import JuMP
import MathOptInterface as MOI

_case(name) = isfile(matpower_case(name)) ? matpower_case(name) : name

# `slacks = true` converts the thermal-limit inequalities into equalities with
# box-bounded slack variables (`p² + q² + s = rate²`, `0 ≤ s ≤ rate²`), so the
# model is equality-constrained + bounds — the form Percival's augmented
# Lagrangian handles directly. `slacks = false` keeps them as `Nonpositives`
# inequalities (matching the ExaModels/GenOpt variable layout).
function build_jump_polar(name::AbstractString; slacks::Bool = false)
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
    for i in 1:nb                # flat voltage start
        JuMP.set_start_value(vm[i], 1.0)
    end
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
    # Thermal limits |S|² ≤ rate² — one vector constraint each.
    if slacks
        JuMP.@variable(model, sf[1:nl], container = ArrayDiff.ArrayOfVariables)
        JuMP.@variable(model, st[1:nl], container = ArrayDiff.ArrayOfVariables)
        _set_bounds(sf, zeros(nl), d.rate_a_sq)
        _set_bounds(st, zeros(nl), d.rate_a_sq)
        for i in 1:nl # start on the constraint: p = q = 0 ⇒ s = rate²
            JuMP.set_start_value(sf[i], d.rate_a_sq[i])
            JuMP.set_start_value(st[i], d.rate_a_sq[i])
        end
        JuMP.@constraint(model, pf .^ 2 .+ qf .^ 2 .+ sf .- d.rate_a_sq in MOI.Zeros(nl))
        JuMP.@constraint(model, pt .^ 2 .+ qt .^ 2 .+ st .- d.rate_a_sq in MOI.Zeros(nl))
    else
        JuMP.@constraint(model, pf .^ 2 .+ qf .^ 2 .- d.rate_a_sq in MOI.Nonpositives(nl))
        JuMP.@constraint(model, pt .^ 2 .+ qt .^ 2 .- d.rate_a_sq in MOI.Nonpositives(nl))
    end
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
