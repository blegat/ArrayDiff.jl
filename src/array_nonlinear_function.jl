"""
    ArrayNonlinearFunction{N} <: MOI.AbstractVectorFunction

Represents an N-dimensional array-valued nonlinear function for MOI.

The `output_dimension` is `prod(size)` — the vectorization of the array — since
`MOI.AbstractVectorFunction` cannot represent multidimensional arrays. No actual
vectorization is performed; this is only for passing through MOI layers.

## Fields

  - `head::Symbol`: the operator (e.g., `:*`, `:tanh`)
  - `args::Vector{Any}`: arguments, which may be `ArrayNonlinearFunction`,
    `MOI.ScalarNonlinearFunction`, `MOI.VariableIndex`, `Float64`,
    `Vector{Float64}`, `Matrix{Float64}`, or `ArrayOfVariableIndices`
  - `size::NTuple{N,Int}`: the dimensions of the output array
  - `broadcasted::Bool`: whether this is a broadcasted operation
"""
struct ArrayNonlinearFunction{N} <: MOI.AbstractVectorFunction
    head::Symbol
    args::Vector{Any}
    size::NTuple{N,Int}
    broadcasted::Bool
end

function MOI.output_dimension(f::ArrayNonlinearFunction)
    return prod(f.size)
end

"""
    ArrayOfVariableIndices{N}

A block of contiguous `MOI.VariableIndex` values representing an N-dimensional
array. Used as an argument in `ArrayNonlinearFunction`.
"""
struct ArrayOfVariableIndices{N} <: MOI.AbstractVectorFunction
    offset::Int
    size::NTuple{N,Int}
end

Base.size(a::ArrayOfVariableIndices) = a.size

function MOI.output_dimension(f::ArrayOfVariableIndices)
    return prod(f.size)
end

function _to_moi(x::ArrayNonlinearFunction)
    return x
end
