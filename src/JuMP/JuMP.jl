# JuMP extension

import JuMP

# Equivalent of `AbstractJuMPScalar` but for arrays
abstract type AbstractJuMPArray{T,N} <: AbstractArray{T,N} end

include("variables.jl")
include("nlp_expr.jl")
include("operators.jl")
include("print.jl")
