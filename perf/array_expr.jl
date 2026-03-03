struct GenericArrayExpr{N,V<:AbstractVariableRef}
    head::Symbol
    args::Vector{Any}
    size::NTuple{N,Int}
end

const ArrayExpr{N} = GenericArrayExpr{N,JuMP.VariableRef}

function LinearAlgebra.mul(A::MatrixOfVariables, B::Matrix)
    return GenericArrayExpr{2,variable_ref_type(A.model)}(:*, Any[A, B], (size(A, 1), size(B, 2)))
end
