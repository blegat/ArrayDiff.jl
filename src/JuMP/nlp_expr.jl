struct GenericArrayExpr{V<:JuMP.AbstractVariableRef,N} <:
       AbstractJuMPArray{JuMP.GenericNonlinearExpr{V},N}
    head::Symbol
    args::Vector{Any}
    size::NTuple{N,Int}
    broadcasted::Bool
end

const GenericMatrixExpr{V<:JuMP.AbstractVariableRef} = GenericArrayExpr{V,2}
const ArrayExpr{N} = GenericArrayExpr{JuMP.VariableRef,N}
const MatrixExpr = ArrayExpr{2}
const VectorExpr = ArrayExpr{1}

function Base.getindex(::GenericArrayExpr, args...)
    return error(
        "`getindex` not implemented, build vectorized expression instead",
    )
end

Base.size(expr::GenericArrayExpr) = expr.size

JuMP.variable_ref_type(::Type{GenericMatrixExpr{V}}) where {V} = V
