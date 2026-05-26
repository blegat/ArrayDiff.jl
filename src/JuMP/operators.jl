function _matmul(::Type{V}, A, B) where {V}
    return GenericMatrixExpr{V}(:*, Any[A, B], (size(A, 1), size(B, 2)), false)
end

function Base.:(*)(A::AbstractJuMPMatrix, B::Matrix)
    return _matmul(JuMP.variable_ref_type(A), A, B)
end
function Base.:(*)(A::Matrix, B::AbstractJuMPMatrix)
    return _matmul(JuMP.variable_ref_type(B), A, B)
end
function Base.:(*)(A::AbstractJuMPMatrix, B::AbstractJuMPMatrix)
    return _matmul(JuMP.variable_ref_type(A), A, B)
end

function __broadcast(
    ::Type{V},
    axes::NTuple{N,Base.OneTo{Int}},
    op::Function,
    args::Vector{Any},
) where {V,N}
    return GenericArrayExpr{V,N}(Symbol(op), args, length.(axes), true)
end

function _broadcast(::Type{V}, op::Function, args...) where {V}
    return __broadcast(V, Broadcast.combine_axes(args...), op, Any[args...])
end

function Base.broadcasted(op::Function, x::AbstractJuMPArray)
    return _broadcast(JuMP.variable_ref_type(x), op, x)
end

function Base.broadcasted(op::Function, x::AbstractJuMPArray, y::AbstractArray)
    return _broadcast(JuMP.variable_ref_type(x), op, x, y)
end

function Base.broadcasted(op::Function, x::AbstractArray, y::AbstractJuMPArray)
    return _broadcast(JuMP.variable_ref_type(y), op, x, y)
end

function Base.broadcasted(
    op::Function,
    x::AbstractJuMPArray,
    y::AbstractJuMPArray,
)
    return _broadcast(JuMP.variable_ref_type(x), op, x, y)
end

function Base.broadcasted(op::Function, x::AbstractJuMPArray, y::Number)
    return _broadcast(JuMP.variable_ref_type(x), op, x, y)
end

function Base.broadcasted(op::Function, x::Number, y::AbstractJuMPArray)
    return _broadcast(JuMP.variable_ref_type(y), op, x, y)
end

function Base.broadcasted(
    ::typeof(Base.literal_pow),
    ::typeof(^),
    x::AbstractJuMPArray,
    ::Val{y},
) where {y}
    return Base.broadcasted(^, x, y)
end

function Base.sum(x::AbstractJuMPArray; dims = Colon())
    V = JuMP.variable_ref_type(x)
    if dims === Colon()
        return JuMP.GenericNonlinearExpr{V}(:sum, Any[x])
    end
    sz = ntuple(i -> i in dims ? 1 : size(x, i), ndims(x))
    dims_vec = JuMP.value_type(V)[d for d in dims]
    return GenericArrayExpr{V,ndims(x)}(:sum_dims, Any[x, dims_vec], sz, false)
end

import LinearAlgebra

function _array_norm(x::AbstractJuMPArray)
    V = JuMP.variable_ref_type(x)
    return JuMP.GenericNonlinearExpr{V}(:norm, Any[x])
end

# Define norm for each concrete AbstractJuMPArray subtype to avoid
# ambiguity with JuMP's error-throwing
#   LinearAlgebra.norm(::AbstractArray{<:AbstractJuMPScalar})
function LinearAlgebra.norm(x::GenericArrayExpr)
    return _array_norm(x)
end

function LinearAlgebra.norm(x::ArrayOfVariables)
    return _array_norm(x)
end

# Subtraction between array expressions and constant arrays
function Base.:(-)(
    x::AbstractJuMPArray{T,N},
    y::AbstractArray{S,N},
) where {S,T,N}
    V = JuMP.variable_ref_type(x)
    @assert size(x) == size(y)
    return GenericArrayExpr{V,N}(:-, Any[x, y], size(x), false)
end

function Base.:(-)(
    x::AbstractArray{S,N},
    y::AbstractJuMPArray{T,N},
) where {S,T,N}
    V = JuMP.variable_ref_type(y)
    @assert size(x) == size(y)
    return GenericArrayExpr{V,N}(:-, Any[x, y], size(y), false)
end

function Base.:(-)(
    x::AbstractJuMPArray{T,N},
    y::AbstractJuMPArray{S,N},
) where {T,S,N}
    V = JuMP.variable_ref_type(x)
    @assert JuMP.variable_ref_type(y) == V
    @assert size(x) == size(y)
    return GenericArrayExpr{V,N}(:-, Any[x, y], size(x), false)
end

# Addition between array expressions and constant arrays
function Base.:(+)(
    x::AbstractJuMPArray{T,N},
    y::AbstractArray{S,N},
) where {S,T,N}
    V = JuMP.variable_ref_type(x)
    @assert size(x) == size(y)
    return GenericArrayExpr{V,N}(:+, Any[x, y], size(x), false)
end

function Base.:(+)(
    x::AbstractArray{S,N},
    y::AbstractJuMPArray{T,N},
) where {S,T,N}
    V = JuMP.variable_ref_type(y)
    @assert size(x) == size(y)
    return GenericArrayExpr{V,N}(:+, Any[x, y], size(y), false)
end

function Base.:(+)(
    x::AbstractJuMPArray{T,N},
    y::AbstractJuMPArray{S,N},
) where {T,S,N}
    V = JuMP.variable_ref_type(x)
    @assert JuMP.variable_ref_type(y) == V
    @assert size(x) == size(y)
    return GenericArrayExpr{V,N}(:+, Any[x, y], size(x), false)
end

# ── User-defined array operators ─────────────────────────────────────────────
#
# `add_operator(f)` wraps `f` in a `JuMP.NonlinearOperator` whose `head` is
# `Symbol(f)`. When the operator is called with at least one `AbstractJuMPArray`
# argument, the dispatch methods below build either a `GenericArrayExpr` (when
# `f` returns an array) or a `JuMP.GenericNonlinearExpr` (scalar output) with
# that `head`. The user is still responsible for registering `f` with the
# `ArrayDiff.Model` via [`UserDefinedArrayOperator`](@ref) so the evaluator can
# call `f` and pull its reverse-mode derivative from `ChainRulesCore.rrule`.

"""
    add_operator(f::Function; head::Symbol = Symbol(f))

Return a `JuMP.NonlinearOperator` wrapping `f`. When the returned operator is
called with `AbstractJuMPArray` arguments, it builds a JuMP expression whose
`head` is `head` — a `GenericArrayExpr` if `f` returns an array, otherwise a
`JuMP.GenericNonlinearExpr`. The output shape is determined by probing `f`
with zero arrays sized like the JuMP-array arguments.
"""
function add_operator(f::Function; head::Symbol = Symbol(f))
    return JuMP.NonlinearOperator(f, head)
end

function _user_op_probe_arg(a::AbstractJuMPArray)
    return zeros(Float64, size(a))
end
_user_op_probe_arg(a::AbstractArray{<:Real}) = Float64.(a)
_user_op_probe_arg(a::Real) = Float64(a)

function _build_user_op_expr(
    op::JuMP.NonlinearOperator,
    V::Type,
    args::Tuple,
)
    probe = map(_user_op_probe_arg, args)
    y = op.func(probe...)
    if y isa AbstractArray
        return GenericArrayExpr{V,ndims(y)}(
            op.head,
            Any[args...],
            size(y),
            false,
        )
    end
    return JuMP.GenericNonlinearExpr{V}(op.head, Any[args...])
end

function (op::JuMP.NonlinearOperator)(x::AbstractJuMPArray)
    V = JuMP.variable_ref_type(x)
    return _build_user_op_expr(op, V, (x,))
end

function (op::JuMP.NonlinearOperator)(
    x::AbstractJuMPArray,
    y::Union{Real,AbstractArray{<:Real},AbstractJuMPArray},
)
    V = JuMP.variable_ref_type(x)
    return _build_user_op_expr(op, V, (x, y))
end

function (op::JuMP.NonlinearOperator)(
    x::Union{Real,AbstractArray{<:Real}},
    y::AbstractJuMPArray,
)
    V = JuMP.variable_ref_type(y)
    return _build_user_op_expr(op, V, (x, y))
end
