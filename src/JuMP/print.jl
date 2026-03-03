function Base.show(io::IO, ::MIME"text/plain", v::ArrayOfVariables)
    return print(io, Base.summary(v), " with offset ", v.offset)
end

function Base.show(io::IO, ::MIME"text/plain", v::GenericArrayExpr)
    return print(io, Base.summary(v))
end

function Base.show(io::IO, v::AbstractJuMPArray)
    return show(io, MIME"text/plain"(), v)
end
