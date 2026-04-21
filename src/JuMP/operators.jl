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
