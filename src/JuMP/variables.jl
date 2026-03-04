# Taken out of GenOpt, we can add ArrayDiff as dependency to GenOpt and remove it in GenOpt

struct ArrayOfVariables{T,N} <: AbstractJuMPArray{JuMP.GenericVariableRef{T},N}
    model::JuMP.GenericModel{T}
    offset::Int64
    size::NTuple{N,Int64}
end

const MatrixOfVariables{T} = ArrayOfVariables{T,2}

Base.size(array::ArrayOfVariables) = array.size
function Base.getindex(A::ArrayOfVariables{T}, I...) where {T}
    index =
        A.offset + Base._to_linear_index(Base.CartesianIndices(A.size), I...)
    return JuMP.GenericVariableRef{T}(A.model, MOI.VariableIndex(index))
end

function JuMP.Containers.container(
    f::Function,
    indices::JuMP.Containers.VectorizedProductIterator{
        NTuple{N,Base.OneTo{Int}},
    },
    ::Type{ArrayOfVariables},
) where {N}
    return to_generator(JuMP.Containers.container(f, indices, Array))
end

JuMP._is_real(::ArrayOfVariables) = true

function Base.convert(
    ::Type{ArrayOfVariables{T,N}},
    array::Array{JuMP.GenericVariableRef{T},N},
) where {T,N}
    model = JuMP.owner_model(array[1])
    offset = JuMP.index(array[1]).value - 1
    for i in eachindex(array)
        @assert JuMP.owner_model(array[i]) === model
        @assert JuMP.index(array[i]).value == offset + i
    end
    return ArrayOfVariables{T,N}(model, offset, size(array))
end

function to_generator(array::Array{JuMP.GenericVariableRef{T},N}) where {T,N}
    return convert(ArrayOfVariables{T,N}, array)
end
