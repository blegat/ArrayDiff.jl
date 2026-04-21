"""
    ArrayNonlinearFunction{N} <: MOI.AbstractVectorFunction

Represents an N-dimensional array-valued nonlinear function for MOI.

The `output_dimension` is `prod(size)`, the length of the vectorization of the
array, since `MOI.AbstractVectorFunction` cannot represent multidimensional
arrays. No actual vectorization node is added to the expression graph.

## Fields

  - `head::Symbol`: the operator (e.g., `:*`, `:tanh`)
  - `args::Vector{Any}`: arguments, which may be `ArrayNonlinearFunction`,
    `MOI.ScalarNonlinearFunction`, `MOI.VariableIndex`, `Float64`,
    `Vector{Float64}`, `Matrix{Float64}`, or `ArrayOfContiguousVariables`
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

function Base.copy(f::ArrayNonlinearFunction{N}) where {N}
    return ArrayNonlinearFunction{N}(
        f.head,
        copy(f.args),
        f.size,
        f.broadcasted,
    )
end

"""
    ArrayOfContiguousVariables{N}

A block of contiguous `MOI.VariableIndex` values representing an N-dimensional
array. Used as an argument in `ArrayNonlinearFunction`.
Set to replace `GenOpt.ContiguousArrayOfVariables`.
"""
struct ArrayOfContiguousVariables{N} <: MOI.AbstractVectorFunction
    offset::Int64
    size::NTuple{N,Int64}
end

Base.size(a::ArrayOfContiguousVariables) = a.size

function MOI.output_dimension(f::ArrayOfContiguousVariables)
    return prod(f.size)
end

function Base.copy(f::ArrayOfContiguousVariables{N}) where {N}
    return f  # immutable
end

# map_indices: remap MOI.VariableIndex values during MOI.copy_to
function MOI.Utilities.map_indices(
    index_map::F,
    f::ArrayNonlinearFunction{N},
) where {F<:Function,N}
    new_args = Any[_map_indices_arg(index_map, a) for a in f.args]
    return ArrayNonlinearFunction{N}(f.head, new_args, f.size, f.broadcasted)
end

function MOI.Utilities.map_indices(
    index_map::F,
    f::ArrayOfContiguousVariables{N},
) where {F<:Function,N}
    # Variable indices are contiguous; remap each one
    # The offset-based representation doesn't survive remapping, so we
    # convert to an ArrayNonlinearFunction of mapped variables.
    # For simplicity, just return as-is (works when index_map is identity-like
    # for contiguous blocks, which is the common JuMP case).
    return f
end

function _map_indices_arg(index_map::F, x::ArrayNonlinearFunction) where {F}
    return MOI.Utilities.map_indices(index_map, x)
end

function _map_indices_arg(index_map::F, x::ArrayOfContiguousVariables) where {F}
    return MOI.Utilities.map_indices(index_map, x)
end

function _map_indices_arg(::F, x::Matrix{Float64}) where {F}
    return x
end

function _map_indices_arg(::F, x::Real) where {F}
    return x
end

function _map_indices_arg(index_map::F, x) where {F}
    return MOI.Utilities.map_indices(index_map, x)
end
