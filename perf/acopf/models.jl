# Vectorized AC-OPF model builders. Both build a JuMP model whose variables
# are contiguous `ArrayDiff.ArrayOfVariables` blocks, express every constraint
# group as one whole-vector residual (matrix-vector products + broadcasts),
# and compile each group into an ArrayDiff residual evaluator.
#
# Inequalities are turned into equalities with box-constrained slacks so the
# first-order AL solver only sees equality residuals + variable bounds:
#   vmin² ≤ |V|² ≤ vmax²  →  Vr² + Vi² - w = 0,        w ∈ [vmin², vmax²]
#   p² + q² ≤ rate²       →  p² + q² + s - rate² = 0,  s ∈ [0, rate²]

import ArrayDiff
import JuMP
import LinearAlgebra
import MathOptInterface as MOI
import SparseArrays

storage_type(::ArrayDiff.Mode{S}) where {S} = S

to_storage(mode, v::Vector{Float64}) = storage_type(mode)(v)

# ── Form 1: rectangular voltages + bus admittance matrix (JuMP tutorial) ────
#
# Variables x = [Vr; Vi; Pg; Qg; w], all length N.
# Residuals:
#   rP = Pg - Pd - (Vr .* Ir + Vi .* Ii) = 0     with I = Y V:
#   rQ = Qg - Qd - (Vi .* Ir - Vr .* Ii) = 0       Ir = G Vr - B Vi
#   rW = Vr.^2 + Vi.^2 - w = 0                     Ii = G Vi + B Vr
# Reference-bus convention of the tutorial: Vi[1] = 0 (bounds), Vr[1] ≥ 0.

function build_rect(
    d::RectData;
    mode = ArrayDiff.Mode{Vector{Float64}}(),
    # Transform applied to the constant admittance components, e.g.
    # `ELLMatrix`, `identity` (SparseMatrixCSC), or `Matrix` (dense tape).
    matrix = ELLMatrix,
    device = identity, # storage transform for structured matrices
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
    obj = sum(d.c2 .* Pg .^ 2 .+ d.c1 .* Pg)
    groups = ResidualGroup[
        ResidualGroup("P-balance", build_residual_evaluator(model, rP, mode), N),
        ResidualGroup("Q-balance", build_residual_evaluator(model, rQ, mode), N),
        ResidualGroup("Vmag", build_residual_evaluator(model, rW, mode), N),
    ]
    obj_ev = build_objective_evaluator(model, obj, mode)
    inf = fill(Inf, N)
    lb_Vr = fill(-d.vmax, N)
    ub_Vr = fill(d.vmax, N)
    lb_Vi = fill(-d.vmax, N)
    ub_Vi = fill(d.vmax, N)
    lb_Vr[1] = 0.0 # tutorial: real(V[1]) ≥ 0
    lb_Vi[1] = ub_Vi[1] = 0.0 # tutorial: imag(V[1]) == 0
    lb = vcat(lb_Vr, lb_Vi, d.Pg_lb, d.Qg_lb, fill(d.vmin^2, N))
    ub = vcat(ub_Vr, ub_Vi, d.Pg_ub, d.Qg_ub, fill(d.vmax^2, N))
    x0 = vcat(
        ones(N),
        zeros(N),
        (d.Pg_lb .+ d.Pg_ub) ./ 2,
        (d.Qg_lb .+ d.Qg_ub) ./ 2,
        ones(N),
    )
    return ALProblem(
        obj_ev,
        1.0 / max(maximum(d.c2), 1.0),
        d.c0,
        groups,
        to_storage(mode, lb),
        to_storage(mode, ub),
        to_storage(mode, x0),
    )
end

# ── Form 2: polar voltages, branch flows with sin/cos (GenOpt/ExaModels) ────
#
# Variables x = [va; vm; pg; qg; pf; pt; qf; qt; sf; st].
# With gather matrices F (branch → from-bus) and T (branch → to-bus),
# Δ = F va - T va, vv = (F vm) .* (T vm), the residual groups are the
# vectorized versions of GenOpt/examples/opf.jl's constraints; the power
# balance uses the transposed gathers (segmented sums) F', T' over branch
# flows and C' over generator injections.

function build_polar(
    d::PolarData;
    mode = ArrayDiff.Mode{Vector{Float64}}(),
    device = identity,
    use_gather::Bool = true,
)
    nb, ng, nl = d.nbus, d.ngen, d.nbranch
    make_gather = if use_gather
        idx -> map_storage(device, GatherMatrix(idx, nb))
    else
        # Sparse baseline (kept by reference on the tape as well). `device`
        # may convert the `SparseMatrixCSC`, e.g. to a `CuSparseMatrixCSR`.
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
    vaf = Fg * va
    vat = Tg * va
    vmf = Fg * vm
    vmt = Tg * vm
    Δ = vaf .- vat
    cΔ = cos.(Δ)
    sΔ = sin.(Δ)
    vv = vmf .* vmt
    # From-side flow definitions (GenOpt's first two branch constraints).
    g1 = pf .- (d.c5 .* vmf .^ 2 .+ d.c3 .* (vv .* cΔ) .+ d.c4 .* (vv .* sΔ))
    g2 =
        qf .+ d.c6 .* vmf .^ 2 .+ d.c4 .* (vv .* cΔ) .- d.c3 .* (vv .* sΔ)
    # To-side flows; cos(-Δ) = cos(Δ), sin(-Δ) = -sin(Δ) folded in.
    g3 = pt .- (d.c7 .* vmt .^ 2 .+ d.c1 .* (vv .* cΔ) .- d.c2 .* (vv .* sΔ))
    g4 =
        qt .+ d.c8 .* vmt .^ 2 .+ d.c2 .* (vv .* cΔ) .+ d.c1 .* (vv .* sΔ)
    # Nodal power balance (scatter branch flows and generation to buses).
    g5 = Ft * pf .+ Tt * pt .- Ct * pg .+ d.pd .+ d.gs .* vm .^ 2
    g6 = Ft * qf .+ Tt * qt .- Ct * qg .+ d.qd .- d.bs .* vm .^ 2
    # Thermal limits |S|² ≤ rate² with slacks in [0, rate²].
    g7 = pf .^ 2 .+ qf .^ 2 .+ sf .- d.rate_a_sq
    g8 = pt .^ 2 .+ qt .^ 2 .+ st .- d.rate_a_sq
    obj = sum(d.cost1 .* pg .^ 2 .+ d.cost2 .* pg)
    groups = ResidualGroup[
        ResidualGroup("pf-def", build_residual_evaluator(model, g1, mode), nl),
        ResidualGroup("qf-def", build_residual_evaluator(model, g2, mode), nl),
        ResidualGroup("pt-def", build_residual_evaluator(model, g3, mode), nl),
        ResidualGroup("qt-def", build_residual_evaluator(model, g4, mode), nl),
        ResidualGroup("P-bal", build_residual_evaluator(model, g5, mode), nb),
        ResidualGroup("Q-bal", build_residual_evaluator(model, g6, mode), nb),
        ResidualGroup("sf-lim", build_residual_evaluator(model, g7, mode), nl),
        ResidualGroup("st-lim", build_residual_evaluator(model, g8, mode), nl),
    ]
    obj_ev = build_objective_evaluator(model, obj, mode)
    lb_va = fill(-Inf, nb)
    ub_va = fill(Inf, nb)
    for r in d.ref_buses
        lb_va[r] = ub_va[r] = 0.0
    end
    lb = vcat(
        lb_va,
        d.vmin,
        d.pmin,
        d.qmin,
        -d.rate_a,
        -d.rate_a,
        -d.rate_a,
        -d.rate_a,
        zeros(nl),
        zeros(nl),
    )
    ub = vcat(
        ub_va,
        d.vmax,
        d.pmax,
        d.qmax,
        d.rate_a,
        d.rate_a,
        d.rate_a,
        d.rate_a,
        d.rate_a_sq,
        d.rate_a_sq,
    )
    x0 = vcat(
        zeros(nb),
        ones(nb),
        (d.pmin .+ d.pmax) ./ 2,
        (d.qmin .+ d.qmax) ./ 2,
        zeros(nl),
        zeros(nl),
        zeros(nl),
        zeros(nl),
        copy(d.rate_a_sq),
        copy(d.rate_a_sq),
    )
    return ALProblem(
        obj_ev,
        1.0 / max(maximum(d.cost1; init = 1.0), maximum(d.cost2; init = 1.0), 1.0),
        sum(d.cost3),
        groups,
        to_storage(mode, lb),
        to_storage(mode, ub),
        to_storage(mode, x0),
    )
end
