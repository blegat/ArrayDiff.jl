# Conversion from JuMP array types to MOI ArrayNonlinearFunction
# and set_objective_function that sets AutomaticDifferentiationBackend.

# ── moi_function: JuMP → MOI ─────────────────────────────────────────────────

function JuMP.moi_function(x::ArrayOfVariables{T,N}) where {T,N}
    return ArrayOfContiguousVariables{N}(x.offset, x.size)
end

function JuMP.moi_function(x::GenericArrayExpr{V,N}) where {V,N}
    args = Any[JuMP.moi_function(a) for a in x.args]
    return ArrayNonlinearFunction{N}(x.head, args, x.size, x.broadcasted)
end

JuMP.moi_function(x::AbstractArray{<:Real}) = x

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

# ── Vector constraints over array expressions ────────────────────────────────
#
# `@constraint(model, expr in set)` where `expr` is an `AbstractJuMPArray`
# (e.g. a vectorized residual `Pg .- Pd .- ...`) and `set` is a vector set
# (`MOI.Zeros`, `MOI.Nonnegatives`, `MOI.Nonpositives`). JuMP's default
# `VectorConstraint` scalarizes the function (`func[idx]` for each index),
# which our array expressions deliberately don't support. Instead we keep the
# expression whole: `moi_function` turns it into a single
# `ArrayNonlinearFunction`, preserving the vectorized structure end-to-end.
#
# Relies on JuMP #3451 (`moi_function`/`_is_real` over `AbstractArray`).

struct _ArrayVectorConstraint{F<:AbstractJuMPArray,S<:MOI.AbstractVectorSet} <:
       JuMP.AbstractConstraint
    func::F
    set::S
end

function JuMP.build_constraint(
    _error::Function,
    func::AbstractJuMPArray,
    set::MOI.AbstractVectorSet,
)
    n = length(func)
    if n != MOI.dimension(set)
        _error(
            "Dimension of the function ($n) does not match the dimension of " *
            "the set ($(MOI.dimension(set))).",
        )
    end
    return _ArrayVectorConstraint(func, set)
end

# `jump_function`/`moi_function`/`moi_set` fall back to the generic
# `AbstractConstraint` methods (which read `.func`/`.set`), so we only need the
# shape and the belongs-to-model check. `moi_function(::GenericArrayExpr)`
# already yields the `ArrayNonlinearFunction`.
JuMP.shape(::_ArrayVectorConstraint) = JuMP.VectorShape()

# The array expression carries whole variable blocks; the per-scalar ownership
# check JuMP does for `Vector`-valued functions doesn't apply. Some JuMP
# versions check the constraint, others the function, so cover both.
JuMP.check_belongs_to_model(::AbstractJuMPArray, ::JuMP.AbstractModel) = nothing
JuMP.check_belongs_to_model(::_ArrayVectorConstraint, ::JuMP.AbstractModel) =
    nothing
