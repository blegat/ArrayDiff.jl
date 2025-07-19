"""
    struct Sizes
        ndims::Vector{Int}
        size_offset::Vector{Int}
        size::Vector{Int}
        storage_offset::Vector{Int}
    end

The node at index `k` is an array of `ndims[k]` dimensions and size `sizes[size_offset[k] .+ (1:ndims[k])]`.
Note that `size_offset` is a nonincreasing vector so that `sizes` can be filled in a forward pass,
which goes through the nodes in decreasing index order.
"""
struct Sizes
    ndims::Vector{Int}
    size_offset::Vector{Int}
    size::Vector{Int}
    storage_offset::Vector{Int}
end

function _size(sizes::Sizes, k::Int, dim::Int)
    return sizes.size[sizes.size_offset[k]+dim]
end

function _size(sizes::Sizes, k::Int)
    return view(sizes.size, sizes.size_offset[k] .+ Base.OneTo(sizes.ndims[k]))
end

function _length(sizes::Sizes, k::Int)
    if sizes.ndims[k] == 0
        return 1
    else
        return prod(_size(sizes, k))
    end
end

_eachindex(sizes::Sizes, k) = Base.OneTo(_length(sizes, k))

_length(sizes::Sizes) = sizes.storage_offset[end]

function _storage_range(sizes::Sizes, k::Int)
    return sizes.storage_offset[k] .+ _eachindex(sizes, k)
end

function _getscalar(x, sizes::Sizes, k::Int)
    return x[sizes.storage_offset[k]+1]
end

function _setscalar!(x, value, sizes::Sizes, k::Int)
    return x[sizes.storage_offset[k]+1] = value
end

function _getindex(x, sizes::Sizes, k::Int, j)
    return x[sizes.storage_offset[k]+j]
end

function _setindex!(x, value, sizes::Sizes, k::Int, j)
    return x[sizes.storage_offset[k]+j] = value
end

"""
    @s(storage[node]) -> _getscalar(storage, f.sizes, node)
    @s(storage[node] = value) -> _setscalar!(storage, value, f.sizes, node)

This "at scalar" converts `getindex` and `setindex!` calls to access the
scalar in a vector corresponding to a node.
"""
macro s(expr)
    if Meta.isexpr(expr, :(=)) && length(expr.args) == 2
        lhs, rhs = expr.args
        @assert Meta.isexpr(lhs, :ref)
        @assert length(expr.args) == 2
        return Expr(
            :call,
            :_setscalar!,
            esc(lhs.args[1]),
            esc(rhs),
            esc(:(f.sizes)),
            esc(lhs.args[2]),
        )
    elseif Meta.isexpr(expr, :ref) && length(expr.args) == 2
        arr, idx = expr.args
        return Expr(
            :call,
            :_getscalar,
            esc(arr),
            esc(:(f.sizes)),
            esc(idx),
        )
    else
        error("Unsupported expression `$expr`")
    end
end


"""
    @j(storage[node]) -> _getindex(storage, f.sizes, node, j)
    @j(storage[node] = value) -> _setindex!(storage, value, f.sizes, node, j)

This "at `j`" converts `getindex` and `setindex!` calls to access
the sub-array in a vector corresponding to a node at its `j`th index.
"""
macro j(expr)
    if Meta.isexpr(expr, :(=)) && length(expr.args) == 2
        lhs, rhs = expr.args
        @assert Meta.isexpr(lhs, :ref)
        @assert length(expr.args) == 2
        return Expr(
            :call,
            :_setindex!,
            esc(lhs.args[1]),
            esc(rhs),
            esc(:(f.sizes)),
            esc(lhs.args[2]),
            esc(:j),
        )
    elseif Meta.isexpr(expr, :ref) && length(expr.args) == 2
        arr, idx = expr.args
        return Expr(
            :call,
            :_getindex,
            esc(arr),
            esc(:(f.sizes)),
            esc(idx),
            esc(:j),
        )
    else
        error("Unsupported expression `$expr`")
    end
end

# /!\ Can only be called in decreasing `k` order
function _add_size!(sizes::Sizes, k::Int, size::Tuple)
    sizes.ndims[k] = length(size)
    sizes.size_offset[k] = length(sizes.size)
    append!(sizes.size, size)
    return
end

function _copy_size!(sizes::Sizes, k::Int, child::Int)
    sizes.ndims[k] = sizes.ndims[child]
    sizes.size_offset[k] = length(sizes.size)
    for i in (sizes.size_offset[child] .+ Base.OneTo(sizes.ndims[child]))
        push!(sizes.size, sizes.size[i])
    end
    return
end

function _assert_scalar_children(sizes, children_arr, children_indices, op)
    for c_idx in children_indices
        @inbounds ix = children_arr[c_idx]
        # We don't support nested vectors of vectors,
        # we only support real numbers and array of real numbers
        @assert sizes.ndims[ix] == 0 "Array argument when expected scalar argument for operator `$op`"
    end
end

function _infer_sizes(
    nodes::Vector{Nonlinear.Node},
    adj::SparseArrays.SparseMatrixCSC{Bool,Int},
)
    sizes = Sizes(
        zeros(Int, length(nodes)),
        zeros(Int, length(nodes)),
        Int[],
        zeros(Int, length(nodes) + 1),
    )
    children_arr = SparseArrays.rowvals(adj)
    for k in length(nodes):-1:1
        node = nodes[k]
        children_indices = SparseArrays.nzrange(adj, k)
        N = length(children_indices)
        if node.type == Nonlinear.NODE_CALL_MULTIVARIATE
            if !(
                node.index in
                eachindex(MOI.Nonlinear.DEFAULT_MULTIVARIATE_OPERATORS)
            )
                # TODO user-defined operators
                continue
            end
            op = MOI.Nonlinear.DEFAULT_MULTIVARIATE_OPERATORS[node.index]
            if op == :vect
                _assert_scalar_children(
                    sizes,
                    children_arr,
                    children_indices,
                    op,
                )
                _add_size!(sizes, k, (N,))
            elseif op == :dot
                # TODO assert all arguments have same size
            elseif op == :+ || op == :-
                # TODO assert all arguments have same size
                _copy_size!(sizes, k, children_arr[first(children_indices)])
            elseif op == :*
                # TODO assert compatible sizes and all ndims should be 0 or 2
                first_matrix = findfirst(children_indices) do i
                    return !iszero(sizes.ndims[children_arr[i]])
                end
                if !isnothing(first_matrix)
                    last_matrix = findfirst(children_indices) do i
                        return !iszero(sizes.ndims[children_arr[i]])
                    end
                    _add_size!(
                        sizes,
                        k,
                        (
                            _size(sizes, first_matrix, 1),
                            _size(sizes, last_matrix, sizes.ndims[last_matrix]),
                        ),
                    )
                end
            elseif op == :^ || op == :/
                @assert N == 2
                _assert_scalar_children(
                    sizes,
                    children_arr,
                    children_indices[2:end],
                    op,
                )
                _copy_size!(sizes, k, children_arr[first(children_indices)])
            else
                _assert_scalar_children(
                    sizes,
                    children_arr,
                    children_indices,
                    op,
                )
            end
        elseif node.type == Nonlinear.NODE_CALL_UNIVARIATE
            if !(
                node.index in
                eachindex(MOI.Nonlinear.DEFAULT_UNIVARIATE_OPERATORS)
            )
                # TODO user-defined operators
                continue
            end
            @assert N == 1
            op = MOI.Nonlinear.DEFAULT_UNIVARIATE_OPERATORS[node.index]
            if op == :+ || op == :-
                _copy_size!(sizes, k, children_arr[first(children_indices)])
            else
                _assert_scalar_children(
                    sizes,
                    children_arr,
                    children_indices,
                    op,
                )
            end
        end
    end
    for k in eachindex(nodes)
        sizes.storage_offset[k+1] = sizes.storage_offset[k] + _length(sizes, k)
    end
    return sizes
end
