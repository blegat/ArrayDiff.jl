function _matmul(::Type{V}, A, B) where {V}
    return GenericMatrixExpr{V}(
        :*,
        Any[A, B],
        (size(A, 1), size(B, 2)),
        false,
    )
end

Base.:(*)(A::AbstractJuMPMatrix, B::Matrix) = _matmul(JuMP.variable_ref_type(A), A, B)
Base.:(*)(A::Matrix, B::AbstractJuMPMatrix) = _matmul(JuMP.variable_ref_type(B), A, B)
Base.:(*)(A::AbstractJuMPMatrix, B::AbstractJuMPMatrix) = _matmul(JuMP.variable_ref_type(A), A, B)

function __broadcast(
    ::Type{V},
    axes::NTuple{N,Base.OneTo{Int}},
    op::Function,
    args::Vector{Any},
) where {V,N}
    return GenericArrayExpr{V,N}(
        Symbol(op),
        args,
        length.(axes),
        true,
    )
end

function _broadcast(
    ::Type{V},
    op::Function,
    args...,
) where {V}
    return __broadcast(
        V,
        Broadcast.combine_axes(args...),
        op,
        Any[args...],
    )
end

function Base.broadcasted(op::Function, x::AbstractJuMPArray)
    return _broadcast(JuMP.variable_ref_type(x), op, x)
end
