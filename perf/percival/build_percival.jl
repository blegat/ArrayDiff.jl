# Build the two AC-OPF forms as `ArrayDiffNLPModel`s (min f s.t. c(x)=0,
# l≤x≤u), reusing the data and structured-matrix helpers from ../acopf.
# Inequalities are converted to equalities with box-constrained slacks, so the
# model has only equality constraints + variable bounds — the form Percival's
# `Val{:equ}` path handles directly.

import ArrayDiff
import JuMP
import LinearAlgebra
import MathOptInterface as MOI
import SparseArrays

const ACOPF_DIR = joinpath(@__DIR__, "..", "acopf")
include(joinpath(ACOPF_DIR, "structured.jl"))
include(joinpath(ACOPF_DIR, "data.jl"))
include("adnlp.jl")

storage_type(::ArrayDiff.Mode{S}) where {S} = S
to_storage(mode, v::Vector{Float64}) = storage_type(mode)(v)

# Build an ArrayDiff residual evaluator for a vector array expression.
function residual_evaluator(model, expr, mode)
    ad = ArrayDiff.model(mode)
    ArrayDiff.set_residual!(ad, JuMP.moi_function(expr))
    ev = ArrayDiff.Evaluator(ad, mode, JuMP.index.(JuMP.all_variables(model)))
    MOI.initialize(ev, Symbol[:Grad, :Jac, :JacVec])
    return ev
end

function objective_evaluator(model, expr, mode)
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(expr))
    ev = ArrayDiff.Evaluator(ad, mode, JuMP.index.(JuMP.all_variables(model)))
    MOI.initialize(ev, Symbol[:Grad])
    return ev
end

# ── Form 1: rectangular voltages + Ybus ──────────────────────────────────────

function build_rect_nlp(
    d::RectData;
    mode = ArrayDiff.Mode{Vector{Float64}}(),
    matrix = ELLMatrix,
    device = identity,
)
    N = d.N
    Gm = map_storage(device, matrix(d.G))
    Bm = map_storage(device, matrix(d.B))
    model = JuMP.Model()
    JuMP.@variable(model, Vr[1:N], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, Vi[1:N], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, Pg[1:N], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, Qg[1:N], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, w[1:N], container = ArrayDiff.ArrayOfVariables)
    Ir = Gm * Vr .- Bm * Vi
    Ii = Gm * Vi .+ Bm * Vr
    rP = Pg .- d.Pd .- (Vr .* Ir .+ Vi .* Ii)
    rQ = Qg .- d.Qd .- (Vi .* Ir .- Vr .* Ii)
    rW = Vr .^ 2 .+ Vi .^ 2 .- w
    obj_expr = sum(d.c2 .* Pg .^ 2 .+ d.c1 .* Pg)
    obj = objective_evaluator(model, obj_expr, mode)
    cons = [
        residual_evaluator(model, rP, mode),
        residual_evaluator(model, rQ, mode),
        residual_evaluator(model, rW, mode),
    ]
    lb_Vr = fill(-d.vmax, N)
    ub_Vr = fill(d.vmax, N)
    lb_Vi = fill(-d.vmax, N)
    ub_Vi = fill(d.vmax, N)
    lb_Vr[1] = 0.0
    lb_Vi[1] = ub_Vi[1] = 0.0
    lb = vcat(lb_Vr, lb_Vi, d.Pg_lb, d.Qg_lb, fill(d.vmin^2, N))
    ub = vcat(ub_Vr, ub_Vi, d.Pg_ub, d.Qg_ub, fill(d.vmax^2, N))
    x0 = vcat(
        ones(N),
        zeros(N),
        (d.Pg_lb .+ d.Pg_ub) ./ 2,
        (d.Qg_lb .+ d.Qg_ub) ./ 2,
        ones(N),
    )
    nvar = 5N
    return ArrayDiffNLPModel(
        obj,
        cons,
        nvar,
        to_storage(mode, lb),
        to_storage(mode, ub),
        to_storage(mode, x0);
        name = "acopf-rect",
    ),
    d.c0 # objective offset (constant term), added back when reporting
end

# ── Form 2: polar voltages, sin/cos branch flows ─────────────────────────────

function build_polar_nlp(
    d::PolarData;
    mode = ArrayDiff.Mode{Vector{Float64}}(),
    device = identity,
    use_gather::Bool = true,
)
    nb, ng, nl = d.nbus, d.ngen, d.nbranch
    make_gather = if use_gather
        idx -> map_storage(device, GatherMatrix(idx, nb))
    else
        idx -> device(
            SparseArrays.sparse(1:length(idx), idx, 1.0, length(idx), nb),
        )
    end
    Fg = make_gather(d.f_bus)
    Tg = make_gather(d.t_bus)
    Cg = make_gather(d.gen_bus)
    Ft = LinearAlgebra.transpose(Fg)
    Tt = LinearAlgebra.transpose(Tg)
    Ct = LinearAlgebra.transpose(Cg)
    model = JuMP.Model()
    JuMP.@variable(model, va[1:nb], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, vm[1:nb], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, pg[1:ng], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, qg[1:ng], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, pf[1:nl], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, pt[1:nl], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, qf[1:nl], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, qt[1:nl], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, sf[1:nl], container = ArrayDiff.ArrayOfVariables)
    JuMP.@variable(model, st[1:nl], container = ArrayDiff.ArrayOfVariables)
    vmf = Fg * vm
    vmt = Tg * vm
    Δ = Fg * va .- Tg * va
    cΔ = cos.(Δ)
    sΔ = sin.(Δ)
    vv = vmf .* vmt
    g1 = pf .- (d.c5 .* vmf .^ 2 .+ d.c3 .* (vv .* cΔ) .+ d.c4 .* (vv .* sΔ))
    g2 = qf .+ d.c6 .* vmf .^ 2 .+ d.c4 .* (vv .* cΔ) .- d.c3 .* (vv .* sΔ)
    g3 = pt .- (d.c7 .* vmt .^ 2 .+ d.c1 .* (vv .* cΔ) .- d.c2 .* (vv .* sΔ))
    g4 = qt .+ d.c8 .* vmt .^ 2 .+ d.c2 .* (vv .* cΔ) .+ d.c1 .* (vv .* sΔ)
    g5 = Ft * pf .+ Tt * pt .- Ct * pg .+ d.pd .+ d.gs .* vm .^ 2
    g6 = Ft * qf .+ Tt * qt .- Ct * qg .+ d.qd .- d.bs .* vm .^ 2
    g7 = pf .^ 2 .+ qf .^ 2 .+ sf .- d.rate_a_sq
    g8 = pt .^ 2 .+ qt .^ 2 .+ st .- d.rate_a_sq
    obj_expr = sum(d.cost1 .* pg .^ 2 .+ d.cost2 .* pg)
    obj = objective_evaluator(model, obj_expr, mode)
    cons = [
        residual_evaluator(model, g, mode) for
        g in (g1, g2, g3, g4, g5, g6, g7, g8)
    ]
    lb_va = fill(-Inf, nb)
    ub_va = fill(Inf, nb)
    for r in d.ref_buses
        lb_va[r] = ub_va[r] = 0.0
    end
    lb = vcat(
        lb_va, d.vmin, d.pmin, d.qmin,
        -d.rate_a, -d.rate_a, -d.rate_a, -d.rate_a, zeros(nl), zeros(nl),
    )
    ub = vcat(
        ub_va, d.vmax, d.pmax, d.qmax,
        d.rate_a, d.rate_a, d.rate_a, d.rate_a, d.rate_a_sq, d.rate_a_sq,
    )
    x0 = vcat(
        zeros(nb), ones(nb), (d.pmin .+ d.pmax) ./ 2, (d.qmin .+ d.qmax) ./ 2,
        zeros(nl), zeros(nl), zeros(nl), zeros(nl),
        copy(d.rate_a_sq), copy(d.rate_a_sq),
    )
    nvar = 2nb + 2ng + 6nl
    return ArrayDiffNLPModel(
        obj,
        cons,
        nvar,
        to_storage(mode, lb),
        to_storage(mode, ub),
        to_storage(mode, x0);
        name = "acopf-polar",
    ),
    sum(d.cost3)
end
