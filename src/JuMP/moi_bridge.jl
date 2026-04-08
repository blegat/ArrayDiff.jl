# Conversion from JuMP array types to MOI ArrayNonlinearFunction
# and set_objective_function that sets AutomaticDifferentiationBackend.

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

# ── Detect whether a JuMP expression contains array args ─────────────────────

_has_array_args(::Any) = false
_has_array_args(::AbstractJuMPArray) = true

function _has_array_args(x::JuMP.GenericNonlinearExpr)
    return any(_has_array_args, x.args)
end

# ── set_objective_function for nonlinear expressions with array args ─────────
# When the expression contains array subexpressions, we set
# AutomaticDifferentiationBackend to ArrayDiff.Mode() so the solver
# creates an ArrayDiff.Model (via nonlinear_model) for parsing.

function JuMP.set_objective_function(
    model::JuMP.GenericModel{T},
    func::JuMP.GenericNonlinearExpr{JuMP.GenericVariableRef{T}},
) where {T<:Real}
    if _has_array_args(func)
        MOI.set(
            JuMP.backend(model),
            MOI.AutomaticDifferentiationBackend(),
            Mode(),
        )
    end
    # Standard JuMP flow: convert to MOI and set on backend
    f = JuMP.moi_function(func)
    attr = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(JuMP.backend(model), attr, f)
    model.is_model_dirty = true
    return
end
