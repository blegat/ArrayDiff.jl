# Conversion from JuMP array types to MOI ArrayNonlinearFunction
# and set_objective_function for scalar-shaped (0-dim) array expressions.

# ── moi_function: JuMP → MOI ─────────────────────────────────────────────────

function _to_moi_arg(x::ArrayOfVariables{T,N}) where {T,N}
    return ArrayOfVariableIndices{N}(x.offset, x.size)
end

function _to_moi_arg(x::GenericArrayExpr{V,N}) where {V,N}
    args = Any[_to_moi_arg(a) for a in x.args]
    return ArrayNonlinearFunction{N}(x.head, args, x.size, x.broadcasted)
end

_to_moi_arg(x::Matrix{Float64}) = x

_to_moi_arg(x::Real) = Float64(x)

function JuMP.moi_function(x::GenericArrayExpr{V,N}) where {V,N}
    return _to_moi_arg(x)
end

# ── set_objective_function for scalar-shaped array expressions ───────────────
# GenericArrayExpr{V,0} (size=()) is scalar-valued but contains array
# subexpressions.  JuMP's default set_objective_function only handles
# AbstractJuMPScalar, so we add a method here.  We also set the
# AutomaticDifferentiationBackend to ArrayDiff.Mode() so that the solver
# uses ArrayDiff's evaluator.

function JuMP.set_objective_function(
    model::JuMP.GenericModel{T},
    func::GenericArrayExpr{JuMP.GenericVariableRef{T},0},
) where {T<:Real}
    f = JuMP.moi_function(func)
    MOI.set(
        JuMP.backend(model),
        MOI.AutomaticDifferentiationBackend(),
        Mode(),
    )
    attr = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(JuMP.backend(model), attr, f)
    model.is_model_dirty = true
    return
end
