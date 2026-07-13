# Data preparation for the two vectorized AC-OPF forms.

import LinearAlgebra
import PowerModels
import SparseArrays

# ── Form 1: the JuMP tutorial's case9mod, rectangular voltages + Ybus ────────
#
# Data copied from JuMP/docs/src/tutorials/applications/optimal_power_flow.jl
# but converted to per-unit so every decision variable is O(1) (essential for
# a first-order method). The objective is kept in dollars by rescaling the
# cost coefficients, so the reference value 3087.84 from the tutorial still
# applies.

struct RectData
    N::Int
    G::SparseArrays.SparseMatrixCSC{Float64,Int} # real(Y) in per-unit
    B::SparseArrays.SparseMatrixCSC{Float64,Int} # imag(Y) in per-unit
    Pg_lb::Vector{Float64} # per-unit bounds on real generation
    Pg_ub::Vector{Float64}
    Qg_lb::Vector{Float64}
    Qg_ub::Vector{Float64}
    Pd::Vector{Float64} # per-unit demands
    Qd::Vector{Float64}
    vmin::Float64
    vmax::Float64
    c2::Vector{Float64} # $ / (p.u.)² etc. so the objective is in dollars
    c1::Vector{Float64}
    c0::Float64
end

function case9mod()
    N = 9
    base_MVA = 100.0
    sv(I, V) = Vector(SparseArrays.sparsevec(I, Float64.(V), N))
    Pg_lb = sv([1, 2, 3], [10, 10, 10]) ./ base_MVA
    Pg_ub = sv([1, 2, 3], [250, 300, 270]) ./ base_MVA
    Qg_lb = sv([1, 2, 3], [-5, -5, -5]) ./ base_MVA
    Qg_ub = sv([1, 2, 3], [300, 300, 300]) ./ base_MVA
    Pd = sv([5, 7, 9], [54, 60, 75]) ./ base_MVA
    Qd = sv([5, 7, 9], [18, 21, 30]) ./ base_MVA
    branch = [
        (1, 4, 0.0, 0.0576, 0.0),
        (4, 5, 0.017, 0.092, 0.158),
        (6, 5, 0.039, 0.17, 0.358),
        (3, 6, 0.0, 0.0586, 0.0),
        (6, 7, 0.0119, 0.1008, 0.209),
        (8, 7, 0.0085, 0.072, 0.149),
        (2, 8, 0.0, 0.0625, 0.0),
        (8, 9, 0.032, 0.161, 0.306),
        (4, 9, 0.01, 0.085, 0.176),
    ]
    M = length(branch)
    F = [b[1] for b in branch]
    T = [b[2] for b in branch]
    # Everything in per-unit here (the tutorial divides z by base_MVA to work
    # in MW/MVar instead).
    z = [b[3] + im * b[4] for b in branch]
    A =
        SparseArrays.sparse(F, 1:M, 1.0, N, M) +
        SparseArrays.sparse(T, 1:M, -1.0, N, M)
    Y_0 = A * SparseArrays.spdiagm(1 ./ z) * A'
    y_sh = [im * b[5] / 2 for b in branch]
    Y_sh = SparseArrays.spdiagm(
        LinearAlgebra.diag(A * SparseArrays.spdiagm(y_sh) * A'),
    )
    Y = Y_0 + Y_sh
    # Tutorial costs are in MW: 0.11 P² + 5 P + 150 etc. With P = base * p:
    c2 = [0.11, 0.085, 0.1225] .* base_MVA^2
    c1 = [5.0, 1.2, 1.0] .* base_MVA
    c0 = 150.0 + 600.0 + 335.0
    # Pad cost vectors to bus length (non-generator buses have Pg = 0 anyway
    # because their bounds are [0, 0]).
    c2v = zeros(N)
    c1v = zeros(N)
    c2v[1:3] .= c2
    c1v[1:3] .= c1
    return RectData(
        N,
        SparseArrays.sparse(real.(Y)),
        SparseArrays.sparse(imag.(Y)),
        Pg_lb,
        Pg_ub,
        Qg_lb,
        Qg_ub,
        Pd,
        Qd,
        0.9,
        1.1,
        c2v,
        c1v,
        c0,
    )
end

# ── Form 2: PowerModels-derived branch data (ExaModels' parametrization) ────
#
# Same `c1..c8` constants as ExaModels' OPF example / GenOpt's examples/opf.jl,
# but assembled into flat vectors indexed by branch / bus / gen, ready for the
# vectorized model.

struct PolarData
    nbus::Int
    ngen::Int
    nbranch::Int
    f_bus::Vector{Int}
    t_bus::Vector{Int}
    gen_bus::Vector{Int}
    ref_buses::Vector{Int}
    c1::Vector{Float64}
    c2::Vector{Float64}
    c3::Vector{Float64}
    c4::Vector{Float64}
    c5::Vector{Float64}
    c6::Vector{Float64}
    c7::Vector{Float64}
    c8::Vector{Float64}
    rate_a::Vector{Float64}
    rate_a_sq::Vector{Float64}
    pd::Vector{Float64}
    qd::Vector{Float64}
    gs::Vector{Float64}
    bs::Vector{Float64}
    vmin::Vector{Float64}
    vmax::Vector{Float64}
    pmin::Vector{Float64}
    pmax::Vector{Float64}
    qmin::Vector{Float64}
    qmax::Vector{Float64}
    cost1::Vector{Float64}
    cost2::Vector{Float64}
    cost3::Vector{Float64}
end

function parse_polar_case(filename::AbstractString)
    data = PowerModels.parse_file(filename)
    PowerModels.standardize_cost_terms!(data; order = 2)
    PowerModels.calc_thermal_limits!(data)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    busdict = Dict(k => i for (i, (k, v)) in enumerate(ref[:bus]))
    gendict = Dict(k => i for (i, (k, v)) in enumerate(ref[:gen]))
    branchdict = Dict(k => i for (i, (k, v)) in enumerate(ref[:branch]))
    nbus = length(ref[:bus])
    ngen = length(ref[:gen])
    nbranch = length(ref[:branch])
    pd = zeros(nbus)
    qd = zeros(nbus)
    gs = zeros(nbus)
    bs = zeros(nbus)
    vmin = zeros(nbus)
    vmax = zeros(nbus)
    for (k, v) in ref[:bus]
        i = busdict[k]
        loads = [ref[:load][l] for l in ref[:bus_loads][k]]
        shunts = [ref[:shunt][s] for s in ref[:bus_shunts][k]]
        pd[i] = sum(load["pd"] for load in loads; init = 0.0)
        qd[i] = sum(load["qd"] for load in loads; init = 0.0)
        gs[i] = sum(shunt["gs"] for shunt in shunts; init = 0.0)
        bs[i] = sum(shunt["bs"] for shunt in shunts; init = 0.0)
        vmin[i] = v["vmin"]
        vmax[i] = v["vmax"]
    end
    gen_bus = zeros(Int, ngen)
    pmin = zeros(ngen)
    pmax = zeros(ngen)
    qmin = zeros(ngen)
    qmax = zeros(ngen)
    cost1 = zeros(ngen)
    cost2 = zeros(ngen)
    cost3 = zeros(ngen)
    for (k, v) in ref[:gen]
        i = gendict[k]
        gen_bus[i] = busdict[v["gen_bus"]]
        pmin[i] = v["pmin"]
        pmax[i] = v["pmax"]
        qmin[i] = v["qmin"]
        qmax[i] = v["qmax"]
        cost1[i] = v["cost"][1]
        cost2[i] = v["cost"][2]
        cost3[i] = v["cost"][3]
    end
    f_bus = zeros(Int, nbranch)
    t_bus = zeros(Int, nbranch)
    c = [zeros(nbranch) for _ in 1:8]
    rate_a = zeros(nbranch)
    for (k, branch) in ref[:branch]
        i = branchdict[k]
        f_bus[i] = busdict[branch["f_bus"]]
        t_bus[i] = busdict[branch["t_bus"]]
        g, b = PowerModels.calc_branch_y(branch)
        tr, ti = PowerModels.calc_branch_t(branch)
        ttm = tr^2 + ti^2
        g_fr = branch["g_fr"]
        b_fr = branch["b_fr"]
        g_to = branch["g_to"]
        b_to = branch["b_to"]
        c[1][i] = (-g * tr - b * ti) / ttm
        c[2][i] = (-b * tr + g * ti) / ttm
        c[3][i] = (-g * tr + b * ti) / ttm
        c[4][i] = (-b * tr - g * ti) / ttm
        c[5][i] = (g + g_fr) / ttm
        c[6][i] = (b + b_fr) / ttm
        c[7][i] = g + g_to
        c[8][i] = b + b_to
        rate_a[i] = branch["rate_a"]
    end
    ref_buses = [busdict[k] for (k, _) in ref[:ref_buses]]
    return PolarData(
        nbus,
        ngen,
        nbranch,
        f_bus,
        t_bus,
        gen_bus,
        ref_buses,
        c[1],
        c[2],
        c[3],
        c[4],
        c[5],
        c[6],
        c[7],
        c[8],
        rate_a,
        rate_a .^ 2,
        pd,
        qd,
        gs,
        bs,
        vmin,
        vmax,
        pmin,
        pmax,
        qmin,
        qmax,
        cost1,
        cost2,
        cost3,
    )
end

matpower_case(name::AbstractString) = joinpath(
    dirname(dirname(pathof(PowerModels))),
    "test",
    "data",
    "matpower",
    name,
)
