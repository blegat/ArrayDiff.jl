# A first-order augmented-Lagrangian solver operating purely through
# ArrayDiff's residual API. Everything in the hot loop is either a
# whole-vector broadcast or an ArrayDiff evaluator call, so the same code
# runs with `Vector{Float64}` (CPU), `JLArray` (GPU-semantics check) and
# `CuVector{Float64}` (GPU) storage.
#
#   min f(x)   s.t.  F_g(x) = 0 (g = 1..#groups),   lb ≤ x ≤ ub
#
# Inequalities are pre-converted to equalities with box-constrained slacks by
# the model builders, so only equality residual groups appear here. Bounds
# are enforced by projection (clamp) inside a projected-Adam inner loop.

import ArrayDiff
import JuMP
import LinearAlgebra
import MathOptInterface as MOI
import Printf

struct ResidualGroup{E}
    name::String
    evaluator::E
    dim::Int
end

struct ALProblem{S<:AbstractVector{Float64},O}
    objective::O # ArrayDiff.Evaluator or nothing
    # The AL minimizes `objective_scale * f(x) + multiplier/penalty terms`;
    # scale down large cost coefficients so the objective gradient is O(1)
    # like the constraint gradients. Reported objective values are unscaled.
    objective_scale::Float64
    # Constant term left out of the evaluator (keeps the tape free of scalar
    # +'s, which are not GPU-safe); added back when reporting.
    objective_offset::Float64
    groups::Vector{ResidualGroup}
    lb::S
    ub::S
    x0::S
end

n_variables(p::ALProblem) = length(p.x0)

"""
    build_residual_evaluator(jump_model, expr, mode)

Compile the vector-valued array expression `expr` (built from the variables
of `jump_model`) into an `ArrayDiff.Evaluator` supporting `eval_residual!`
and `eval_residual_jtprod!` with tape storage given by `mode`.
"""
function build_residual_evaluator(jump_model, expr, mode)
    ad = ArrayDiff.model(mode)
    ArrayDiff.set_residual!(ad, JuMP.moi_function(expr))
    ev = ArrayDiff.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(jump_model)),
    )
    MOI.initialize(ev, Symbol[:Grad, :Jac, :JacVec])
    return ev
end

function build_objective_evaluator(jump_model, expr, mode)
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(expr))
    ev = ArrayDiff.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(jump_model)),
    )
    MOI.initialize(ev, Symbol[:Grad])
    return ev
end

# Buffers for one residual group, allocated with the problem's storage type.
struct GroupState{S}
    F::S   # residual values
    λ::S   # multipliers
    v::S   # λ + ρ F seed for J'v
    Jtv::S # J' (λ + ρ F)
end

mutable struct ALState{S}
    x::S
    g::S # gradient of the AL
    m::S # Adam first moment / SPG previous x
    v::S # Adam second moment / SPG previous g
    step::S # work buffer
    trial::S # SPG line-search trial point
    groups::Vector{GroupState{S}}
end

function ALState(p::ALProblem{S}) where {S}
    n = n_variables(p)
    zero_n = () -> fill!(similar(p.x0, n), 0.0)
    groups = [
        GroupState{S}(
            fill!(similar(p.x0, g.dim), 0.0),
            fill!(similar(p.x0, g.dim), 0.0),
            fill!(similar(p.x0, g.dim), 0.0),
            zero_n(),
        ) for g in p.groups
    ]
    x = copy(p.x0)
    x .= clamp.(x, p.lb, p.ub)
    return ALState{S}(
        x,
        zero_n(),
        zero_n(),
        zero_n(),
        zero_n(),
        zero_n(),
        groups,
    )
end

objective_value(p::ALProblem{S,Nothing}, x) where {S} = p.objective_offset

function objective_value(p::ALProblem, x)
    return MOI.eval_objective(p.objective, x) + p.objective_offset
end

# ∇(AL) = scale⁻¹-free scaled objective gradient + Σ_g J_g' (λ_g + ρ F_g).
function al_gradient!(st::ALState, p::ALProblem, ρ::Float64)
    if p.objective === nothing
        fill!(st.g, 0.0)
    else
        MOI.eval_objective_gradient(p.objective, st.g, st.x)
        st.g .*= p.objective_scale
    end
    for (grp, gs) in zip(p.groups, st.groups)
        ArrayDiff.eval_residual!(grp.evaluator, gs.F, st.x)
        gs.v .= gs.λ .+ ρ .* gs.F
        ArrayDiff.eval_residual_jtprod!(grp.evaluator, gs.Jtv, st.x, gs.v)
        st.g .+= gs.Jtv
    end
    return
end

function residuals!(st::ALState, p::ALProblem)
    for (grp, gs) in zip(p.groups, st.groups)
        ArrayDiff.eval_residual!(grp.evaluator, gs.F, st.x)
    end
    return
end

# AL value at `x` (with the objective already scaled): uses each group's F
# buffer as scratch.
function al_value(st::ALState, p::ALProblem, x, ρ::Float64)
    val =
        p.objective === nothing ? 0.0 :
        p.objective_scale * MOI.eval_objective(p.objective, x)
    for (grp, gs) in zip(p.groups, st.groups)
        ArrayDiff.eval_residual!(grp.evaluator, gs.F, x)
        val += LinearAlgebra.dot(gs.λ, gs.F) + (ρ / 2) * LinearAlgebra.dot(gs.F, gs.F)
    end
    return val
end

# One inner minimization of the AL with a nonmonotone spectral projected
# gradient (Birgin–Martínez–Raydan SPG). Everything is a broadcast or a
# `dot`, so it is storage-generic like the rest of the solver.
function spg!(
    st::ALState,
    p::ALProblem,
    ρ::Float64;
    iters::Int,
    tol::Float64,
    memory::Int = 10,
    α0::Float64 = 1.0,
    αmin::Float64 = 1e-12,
    αmax::Float64 = 1e12,
    γ::Float64 = 1e-4,
)
    x_prev, g_prev, d, trial = st.m, st.v, st.step, st.trial
    φ = al_value(st, p, st.x, ρ)
    al_gradient!(st, p, ρ)
    recent = fill(φ, memory)
    α = α0
    for it in 1:iters
        # d = P(x - α g) - x
        d .= clamp.(st.x .- α .* st.g, p.lb, p.ub) .- st.x
        dnorm = maximum(abs, d)
        if dnorm < tol
            break
        end
        gd = LinearAlgebra.dot(st.g, d)
        # Nonmonotone Armijo backtracking on λ ∈ (0, 1].
        φ_ref = maximum(recent)
        λ = 1.0
        φ_new = φ
        for _ in 1:30
            trial .= st.x .+ λ .* d
            φ_new = al_value(st, p, trial, ρ)
            if φ_new <= φ_ref + γ * λ * gd
                break
            end
            λ /= 2
        end
        x_prev .= st.x
        g_prev .= st.g
        st.x .= trial
        φ = φ_new
        recent[1+it%memory] = φ
        al_gradient!(st, p, ρ)
        # Barzilai–Borwein step for the next iteration.
        x_prev .= st.x .- x_prev # s
        g_prev .= st.g .- g_prev # y
        sy = LinearAlgebra.dot(x_prev, g_prev)
        ss = LinearAlgebra.dot(x_prev, x_prev)
        α = sy > 0 ? clamp(ss / sy, αmin, αmax) : αmax
    end
    return
end

max_violation(st::ALState) =
    isempty(st.groups) ? 0.0 :
    maximum(gs -> isempty(gs.F) ? 0.0 : maximum(abs, gs.F), st.groups)

"""
    solve!(p::ALProblem; kwargs...) -> (x, stats)

Run the first-order augmented-Lagrangian loop. Key knobs:

* `method`: `:adam` (default) or `:spg` for the inner minimization. Adam's
  slow trajectories reach better local basins from a cold start on these
  nonconvex problems; SPG converges faster but to nearer (often worse)
  stationary points, so it is used by default only in the final `polish`
  phase, which drives the violation down from the point Adam reached.
* `outer`, `inner`: number of outer AL updates and inner steps.
* `lr`: initial Adam step, decayed by `lr_decay` each outer iteration
  (floored at `lr_min`).
* `ρ0`, `ρ_growth`, `ρmax`: penalty schedule. Multipliers are updated only
  when the violation meets the current target `η` (LANCELOT-style
  safeguard, tightened by `viol_target_ratio` after each accepted update);
  otherwise `ρ` grows.
* `polish`, `polish_inner`: number of SPG polish rounds at the end.
* `tol_feas`: stop when the max violation falls below this.
"""
function solve!(
    p::ALProblem;
    method::Symbol = :adam,
    outer::Int = 40,
    inner::Int = 6_000,
    inner_tol0::Float64 = 1e-3,
    inner_tol_decay::Float64 = 0.5,
    lr::Float64 = 5e-3,
    lr_decay::Float64 = 0.93,
    lr_min::Float64 = 1e-5,
    ρ0::Float64 = 10.0,
    ρ_growth::Float64 = 3.0,
    ρmax::Float64 = 1e6,
    viol_target_ratio::Float64 = 0.25,
    tol_feas::Float64 = 1e-7,
    β1::Float64 = 0.9,
    β2::Float64 = 0.999,
    ϵ::Float64 = 1e-8,
    polish::Int = 8,
    polish_inner::Int = 2_000,
    verbose::Bool = true,
)
    st = ALState(p)
    ρ = ρ0
    η = 0.1 # violation target for accepting a multiplier update
    α = lr
    inner_tol = inner_tol0
    history = NamedTuple[]
    for out in 1:outer
        if method === :spg
            spg!(st, p, ρ; iters = inner, tol = inner_tol)
            inner_tol = max(inner_tol * inner_tol_decay, 1e-10)
        elseif method === :adam
            fill!(st.m, 0.0)
            fill!(st.v, 0.0)
            for it in 1:inner
                al_gradient!(st, p, ρ)
                st.m .= β1 .* st.m .+ (1 - β1) .* st.g
                st.v .= β2 .* st.v .+ (1 - β2) .* st.g .* st.g
                c1 = 1 - β1^it
                c2 = 1 - β2^it
                st.step .= (α / c1) .* st.m ./ (sqrt.(st.v ./ c2) .+ ϵ)
                st.x .= clamp.(st.x .- st.step, p.lb, p.ub)
            end
        else
            error("unknown method $method")
        end
        residuals!(st, p)
        viol = max_violation(st)
        obj = objective_value(p, st.x)
        push!(history, (outer = out, viol = viol, obj = obj, ρ = ρ, lr = α))
        if verbose
            Printf.@printf(
                "outer %3d  obj %14.6f  viol %10.3e  ρ %8.1e  lr %8.1e\n",
                out,
                obj,
                viol,
                ρ,
                α,
            )
        end
        if viol < tol_feas
            break
        end
        # LANCELOT-style safeguarded update: only trust a first-order
        # multiplier update when the violation met the current target η;
        # otherwise increase the penalty and keep the multipliers.
        if viol <= η
            for gs in st.groups
                gs.λ .+= ρ .* gs.F
            end
            η = max(η * viol_target_ratio, tol_feas / 10)
        else
            ρ = min(ρ * ρ_growth, ρmax)
        end
        α = max(α * lr_decay, lr_min)
    end
    # Final polish: from the (near-feasible) point Adam reached, a few SPG
    # solves with multiplier updates drive the violation down much faster
    # than more Adam steps would. SPG alone from a cold start tends to land
    # in worse local basins, so it is only used here at the end.
    for pol in 1:polish
        residuals!(st, p)
        if max_violation(st) < tol_feas
            break
        end
        spg!(st, p, ρ; iters = polish_inner, tol = 1e-10)
        residuals!(st, p)
        viol = max_violation(st)
        obj = objective_value(p, st.x)
        push!(history, (outer = -pol, viol = viol, obj = obj, ρ = ρ, lr = 0.0))
        if verbose
            Printf.@printf(
                "polish %2d  obj %14.6f  viol %10.3e  ρ %8.1e\n",
                pol,
                obj,
                viol,
                ρ,
            )
        end
        for gs in st.groups
            gs.λ .+= ρ .* gs.F
        end
        ρ = min(ρ * ρ_growth, ρmax)
    end
    residuals!(st, p)
    stats = (
        viol = max_violation(st),
        obj = objective_value(p, st.x),
        history = history,
    )
    return st.x, stats
end
