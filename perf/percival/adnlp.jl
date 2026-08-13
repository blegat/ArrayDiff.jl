# Constrained NLPModel backed by ArrayDiff evaluators:
#   * `obj`  — scalar objective f(x), via reverse-mode gradient
#   * `cons` — a list of vector residuals c_g(x), each evaluated with the
#     vectorized residual API (eval_residual! / jtprod! / jprod!). They are
#     presented to NLPModels as one stacked equality constraint c(x) = 0.
#
# Solving  min f(x)  s.t.  c(x) = 0,  l ≤ x ≤ u  through the NLPModels
# interface lets any NLPModels solver (here Percival) consume it. Storage-
# generic: the vector type `V` follows the ArrayDiff tape (`Vector`, `JLArray`,
# `CuVector`), so the whole solve runs on-device.
#
# Keeping the groups separate (instead of one `vcat`-ed residual) avoids
# needing n-ary `vcat` on the tape and keeps each group a single fused
# vectorized pass — exactly the structure of the hand-written AL solver.
#
# Prototype: moves into the NLPModelsJuMP ArrayDiff extension once validated.

import ArrayDiff
import MathOptInterface as MOI
import NLPModels

mutable struct ArrayDiffNLPModel{T, V, Ro, Rc} <: NLPModels.AbstractNLPModel{T, V}
    meta::NLPModels.NLPModelMeta{T, V}
    counters::NLPModels.Counters
    obj::ArrayDiff.Evaluator{T, Ro}
    cons::Vector{ArrayDiff.Evaluator{T, Rc}}
    offsets::Vector{Int}   # group g occupies rows offsets[g]+1 : offsets[g+1]
    Jtv_tmp::V             # scratch for accumulating J'v across groups
end

function ArrayDiffNLPModel(
    obj::ArrayDiff.Evaluator{T, Ro},
    cons::Vector{ArrayDiff.Evaluator{T, Rc}},
    nvar::Int,
    lvar::V,
    uvar::V,
    x0::V;
    name::String = "ArrayDiffNLP",
) where {T, V, Ro, Rc}
    dims = [ArrayDiff.residual_dimension(c) for c in cons]
    offsets = cumsum(vcat(0, dims))
    ncon = offsets[end]
    z = fill!(similar(x0, ncon), zero(T))
    meta = NLPModels.NLPModelMeta{T, V}(
        nvar;
        x0 = x0,
        lvar = lvar,
        uvar = uvar,
        ncon = ncon,
        lcon = z,
        ucon = copy(z),
        y0 = copy(z),
        nnzh = 0,   # no exact Hessian; quasi-Newton solvers supply their own
        minimize = true,
        islp = false,
        name = name,
        lin = Int[],
        # `findall`-based bound analysis scalar-indexes GPU bound vectors;
        # the solver only projects with lvar/uvar, so skip it.
        variable_bounds_analysis = false,
        constraint_bounds_analysis = false,
        # No second-order info from ArrayDiff.
        hess_available = false,
        hprod_available = false,
    )
    Jtv_tmp = fill!(similar(x0, nvar), zero(T))
    return ArrayDiffNLPModel{T, V, Ro, Rc}(
        meta,
        NLPModels.Counters(),
        obj,
        cons,
        offsets,
        Jtv_tmp,
    )
end

_slice(nlp::ArrayDiffNLPModel, g::Int) = (nlp.offsets[g] + 1):nlp.offsets[g + 1]

function NLPModels.obj(nlp::ArrayDiffNLPModel, x::AbstractVector)
    NLPModels.increment!(nlp, :neval_obj)
    return MOI.eval_objective(nlp.obj, x)
end

function NLPModels.grad!(nlp::ArrayDiffNLPModel, x::AbstractVector, g::AbstractVector)
    NLPModels.increment!(nlp, :neval_grad)
    MOI.eval_objective_gradient(nlp.obj, g, x)
    return g
end

function NLPModels.cons!(nlp::ArrayDiffNLPModel, x::AbstractVector, c::AbstractVector)
    NLPModels.increment!(nlp, :neval_cons)
    for g in eachindex(nlp.cons)
        ArrayDiff.eval_residual!(nlp.cons[g], view(c, _slice(nlp, g)), x)
    end
    return c
end

# Stacked J = [J_1; …; J_G], so (Jv)_g = J_g v: write each group into its slice.
function NLPModels.jprod!(
    nlp::ArrayDiffNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    NLPModels.increment!(nlp, :neval_jprod)
    for g in eachindex(nlp.cons)
        ArrayDiff.eval_residual_jprod!(nlp.cons[g], view(Jv, _slice(nlp, g)), x, v)
    end
    return Jv
end

# J' v = Σ_g J_g' v_g (v_g the slice of v): accumulate group contributions.
function NLPModels.jtprod!(
    nlp::ArrayDiffNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    NLPModels.increment!(nlp, :neval_jtprod)
    fill!(Jtv, zero(eltype(Jtv)))
    for g in eachindex(nlp.cons)
        vg = collect_slice(v, _slice(nlp, g))
        ArrayDiff.eval_residual_jtprod!(nlp.cons[g], nlp.Jtv_tmp, x, vg)
        Jtv .+= nlp.Jtv_tmp
    end
    return Jtv
end

# `eval_residual_jtprod!` seeds the residual root with `v`; a plain `view` is
# fine on CPU and GPU. Kept as a hook in case a contiguous copy is needed.
collect_slice(v::AbstractVector, r) = view(v, r)

# ── Explicit Jacobian (for interior-point solvers like MadNLP) ────────────────
#
# ArrayDiff is matrix-free (jprod/jtprod), but MadNLP's KKT system needs the
# constraint Jacobian as coordinates. We materialize it densely: row i of J is
# ∇c_i = J' e_i, one reverse pass per constraint row. Fine for the dense KKT
# path on modest problems; large/GPU problems would want a sparse assembly.
# Structure is emitted in column-major dense order (all rows for column 1,
# then column 2, ...), and `jac_coord!` fills the same order.

function NLPModels.jac_structure!(
    nlp::ArrayDiffNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    m, n = nlp.meta.ncon, nlp.meta.nvar
    k = 0
    for j in 1:n, i in 1:m
        k += 1
        rows[k] = i
        cols[k] = j
    end
    return rows, cols
end

function NLPModels.jac_coord!(
    nlp::ArrayDiffNLPModel,
    x::AbstractVector,
    vals::AbstractVector,
)
    NLPModels.increment!(nlp, :neval_jac)
    m, n = nlp.meta.ncon, nlp.meta.nvar
    ei = fill!(similar(x, m), zero(eltype(x)))
    row = similar(x, n)
    valsm = reshape(vals, m, n) # column-major: valsm[i, j] = J[i, j]
    for i in 1:m
        fill!(ei, zero(eltype(x)))
        ei_view = view(ei, i:i)
        ei_view .= one(eltype(x))
        NLPModels.jtprod!(nlp, x, ei, row) # row = ∇c_i = J[i, :]
        valsm[i, :] .= row
    end
    return vals
end

# No exact Hessian (`nnzh == 0`). Quasi-Newton solvers (MadNLP's CompactLBFGS,
# Percival's LBFGSModel) build their own approximation from gradients; these
# no-ops satisfy the interface without ArrayDiff ever computing second order.
function NLPModels.hess_structure!(
    nlp::ArrayDiffNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return rows, cols
end

function NLPModels.hess_coord!(
    nlp::ArrayDiffNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector;
    obj_weight = one(eltype(x)),
)
    NLPModels.increment!(nlp, :neval_hess)
    return vals
end
