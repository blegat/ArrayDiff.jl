function Base.:(*)(A::MatrixOfVariables, B::Matrix)
    return GenericArrayExpr{JuMP.variable_ref_type(A.model),2}(:*, Any[A, B], (size(A, 1), size(B, 2)))
end
