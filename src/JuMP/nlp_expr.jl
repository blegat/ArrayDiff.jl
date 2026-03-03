struct GenericArrayExpr{V<:JuMP.AbstractVariableRef,N} <: AbstractJuMPArray{JuMP.GenericNonlinearExpr{V},N}
    head::Symbol
    args::Vector{Any}
    size::NTuple{N,Int}
end

const ArrayExpr{N} = GenericArrayExpr{JuMP.VariableRef,N}

function Base.getindex(::GenericArrayExpr, args...)
    error("`getindex` not implemented, build vectorized expression instead")
end

Base.size(expr::GenericArrayExpr) = expr.size
