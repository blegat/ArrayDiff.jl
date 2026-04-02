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

function LinearAlgebra.norm(x::AbstractJuMPArray)
    V = JuMP.variable_ref_type(x)
    return JuMP.GenericNonlinearExpr{V}(:norm, Any[x])
end
