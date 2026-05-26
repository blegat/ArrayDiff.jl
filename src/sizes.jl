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
    # Use a 1-element view + broadcast so this works on GPU storage as well as
    # `Vector{Float64}`. Direct `x[idx] = value` is a scalar setindex which
    # GPUArrays disallows by default.
    _view_scalar(x, sizes, k) .= value
    return value
end

"""
    _scalar_pos(sizes, k) -> Int

Tape index of node `k`'s single scalar slot. Useful when callers want to build
a 1-element view onto `forward_storage`/`reverse_storage` to do a
broadcast-style read or write that's safe on a GPU array.
"""
@inline _scalar_pos(sizes::Sizes, k::Int) = sizes.storage_offset[k] + 1

function _getindex(x, sizes::Sizes, k::Int, j)
    return x[sizes.storage_offset[k]+j]
end

function _setindex!(x, value, sizes::Sizes, k::Int, j)
    return x[sizes.storage_offset[k]+j] = value
end

"""
    _scalar_load(storage, idx) -> Float64

Read a single Float64 from `storage` at linear index `idx`. The default
implementation just calls `getindex`; this is a hook for storage backends
(such as `CuVector`) that disallow scalar indexing and need to dispatch to a
1-element transfer instead.
"""
_scalar_load(storage::AbstractVector, idx::Int) = @inbounds storage[idx]

function _view_scalar(storage::AbstractVector, sizes::Sizes, k::Int)
    pos = _scalar_pos(sizes, k)
    return view(storage, reshape(pos:pos, ()))
end

"""
    _view_linear(storage, sizes, k) -> SubArray

Return a flat 1-D view of the slice of `storage` that holds node `k`'s array
value. The view aliases the underlying `storage` (no copy), so mutating it
writes back into the tape. For a scalar (`ndims[k] == 0`) node this returns
a length-1 vector view.

Use this for elementwise (broadcasted) operations and reductions that don't
need the array's natural shape — keeping the return type-stable
(`SubArray{T,1,...}`) avoids the heap-boxing that a multi-shape return type
would force.
"""
function _view_linear(storage::AbstractVector, sizes::Sizes, k::Int)
    offset = sizes.storage_offset[k]
    N = _length(sizes, k)
    return view(storage, (offset+1):(offset+N))
end

"""
    _view_matrix(storage, sizes, k) -> ReshapedArray

Return a 2-D view of the slice of `storage` that holds node `k`'s array
value. A 1-D node is treated as a column vector `(n, 1)` and a 0-D node as
`(1, 1)`. Always returns a 2-D `Base.ReshapedArray`, which is what callers
like `LinearAlgebra.mul!` need; keeping the return type-stable avoids
heap-boxing.
"""
function _view_matrix(storage::AbstractVector, sizes::Sizes, k::Int)
    @assert sizes.ndims[k] == 2
    offset = sizes.storage_offset[k]
    size_off = sizes.size_offset[k]
    m = sizes.size[size_off+1]
    n = sizes.size[size_off+2]
    v = view(storage, (offset+1):(offset+m*n))
    return reshape(v, (m, n))
end

# Force specialization, (args..., x) is calling Core._apply_iterate
@inline _push(::Tuple{}, x) = (x,)
@inline _push(t::Tuple{Any}, x) = (t[1], x)
@inline _push(t::Tuple{Any,Any}, x) = (t[1], t[2], x)
@inline _push(t::Tuple{Any,Any,Any}, x) = (t[1], t[2], t[3], x)
@inline _push(t::Tuple{Any,Any,Any,Any}, x) = (t[1], t[2], t[3], t[4], x)
@inline _push(t::Tuple{Any,Any,Any,Any,Any}, x) =
    (t[1], t[2], t[3], t[4], t[5], x)
@inline _push(t::Tuple{Any,Any,Any,Any,Any,Any}, x) =
    (t[1], t[2], t[3], t[4], t[5], t[6], x)
@inline _push(t::Tuple{Any,Any,Any,Any,Any,Any,Any}, x) =
    (t[1], t[2], t[3], t[4], t[5], t[6], t[7], x)

# The following is almost allocation-free but I couldn't figure out why it's not
# Claude generated below an implementation with a generated function but it if we could
# make the simpler version, without @generated, work, it would be simpler.

#@inline function _reshape_call(
#    _,
#    _::Sizes,
#    ::Tuple{},
#    op::F,
#    args::Tuple,
#) where {F<:Function}
#    op(args...)
#    return
#end

#@inline function _reshape_call(
#    storage,
#    sizes::Sizes,
#    nodes::NTuple{N,Int},
#    op::F,
#    args::Tuple,
#) where {N,F<:Function}
#    node = first(nodes)
#    ndims = sizes.ndims[node]
#    if ndims == 0
#        _reshape_call(
#            storage,
#            sizes,
#            Base.tail(nodes),
#            op,
#            _push(args, _view_scalar(storage, sizes, node)),
#        )
#    elseif ndims == 1
#        _reshape_call(
#            storage,
#            sizes,
#            Base.tail(nodes),
#            op,
#            _push(args, _view_linear(storage, sizes, node)),
#        )
#    elseif ndims == 2
#        _reshape_call(
#            storage,
#            sizes,
#            Base.tail(nodes),
#            op,
#            _push(args, _view_matrix(storage, sizes, node)),
#        )
#    else
#        @assert false
#        #        _reshape_call(
#        #            storage,
#        #            sizes,
#        #            Base.tail(nodes),
#        #            args...,
#        #            _view_array(storage, sizes, node), # TODO
#        #        )
#    end
#    return
#end

@generated function _reshape_call(
    storage,
    sizes::Sizes,
    nodes::NTuple{N,Int},
    op::F,
    args::Tuple = (),
) where {N,F}
    # At each level, branch on ndims and CONTINUE recursively inside the chosen
    # branch with the concrete view type. No phi-merge into Any, no Union args.
    # The price: 4^N specialized paths, but each is fully type-stable.
    function emit_level(i, args_sym)
        if i > N
            return Expr(:call, :op, Expr(:..., args_sym))
        end
        node_i = Symbol(:node_, i)
        nd_i = Symbol(:nd_, i)
        # In each branch, push the concrete view onto args and recurse to next level.
        function with_view(view_call)
            v = Symbol(:view_, i)
            a = Symbol(:args_, i)
            return Expr(
                :block,
                :($v = $view_call),
                :($a = _push($args_sym, $v)),
                emit_level(i + 1, a),
            )
        end
        return Expr(
            :block,
            :($node_i = nodes[$i]),
            :($nd_i = sizes.ndims[$node_i]),
            Expr(
                :if,
                :($nd_i == 0),
                with_view(:(_view_scalar(storage, sizes, $node_i))),
                Expr(
                    :elseif,
                    :($nd_i == 1),
                    with_view(:(_view_linear(storage, sizes, $node_i))),
                    Expr(
                        :elseif,
                        :($nd_i == 2),
                        with_view(:(_view_matrix(storage, sizes, $node_i))),
                        with_view(:(_view_array(storage, sizes, $node_i))),
                    ),
                ),
            ),
        )
    end
    return Expr(:block, emit_level(1, :args), :(return nothing))
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
        return Expr(:call, :_getscalar, esc(arr), esc(:(f.sizes)), esc(idx))
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
function _add_size!(sizes::Sizes, k::Int, size)
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

"""
    infer_sizes(op, child_sizes::Tuple...) -> Tuple

Return the output shape of applying `op` to arguments of shapes `child_sizes`.
Each `child_sizes[i]` is `()` if argument `i` is a scalar, or a tuple of
positive integers if it is an array. The returned shape is `()` for a scalar
result.

The default implementation constructs dummy arguments with `zeros(sz)`
(or `0.0` for scalars) and calls `op(args...)`. Specialise on `op`'s
`typeof` to avoid the allocation, to support operators that error on zero
inputs, or to compute the output shape symbolically.

## Example

For multiplication, `infer_sizes` can be implemented as follows
```julia
function infer_sizes(::typeof(*), lhs, rhs)
    if isempty(lhs)
        return rhs
    end
    if isempty(rhs)
        return lhs
    end
    return (lhs[1:end-1]..., rhs[2:end]...)
end
```
"""
"""
function infer_sizes(op, child_sizes::Tuple...)
    args = map(child_sizes) do sz
        return isempty(sz) ? 0.0 : zeros(sz)
    end
    y = op(args...)
    return y isa AbstractArray ? size(y) : ()
end

function _infer_sizes(
    nodes::Vector{Node},
    adj::SparseArrays.SparseMatrixCSC{Bool,Int},
    block_shapes::Dict{Int,Vector{Int}},
    const_values::AbstractVector, # Needed for sum(_; dims)
    operators,
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
        # Block leaves carry their shape in `block_shapes`; they're 2D (m, n)
        # by construction.
        if node.type == NODE_VARIABLE_BLOCK ||
           node.type == NODE_VALUE_BLOCK ||
           node.type == NODE_MOI_VARIABLE_BLOCK
            shape = block_shapes[k]
            _add_size!(sizes, k, shape)
            continue
        end
        children_indices = SparseArrays.nzrange(adj, k)
        N = length(children_indices)
        if node.type == NODE_CALL_MULTIVARIATE
            if !(node.index in eachindex(DEFAULT_MULTIVARIATE_OPERATORS))
                if operators !== nothing &&
                   node.index in eachindex(operators.multivariate_operators)
                    op_sym = operators.multivariate_operators[node.index]
                    if haskey(operators.chainrules_operators, op_sym)
                        f = operators.chainrules_operators[op_sym]
                        child_shapes = Tuple(
                            ntuple(
                                d -> _size(sizes, children_arr[c_idx], d),
                                sizes.ndims[children_arr[c_idx]],
                            ) for c_idx in children_indices
                        )
                        out_sz = infer_sizes(Float64, f, child_shapes...)
                        if !isempty(out_sz)
                            _add_size!(sizes, k, out_sz)
                        end
                        # Scalar output → ndims = 0 (already initialised).
                        continue
                    end
                end
                # TODO user-defined operators
                continue
            end
            op = DEFAULT_MULTIVARIATE_OPERATORS[node.index]
            if op == :vect
                _assert_scalar_children(
                    sizes,
                    children_arr,
                    children_indices,
                    op,
                )
                _add_size!(sizes, k, (N,))
            elseif op == :row
                _assert_scalar_children(
                    sizes,
                    children_arr,
                    children_indices,
                    op,
                )
                _add_size!(sizes, k, (1, N))
            elseif op == :dot
                # TODO assert all arguments have same size
            elseif op == :norm
                # TODO actually norm should be moved to univariate
            elseif op == :sum
                # sum reduces array to scalar, ndims stays 0
            elseif op == :+ || op == :-
                # TODO assert all arguments have same size
                _copy_size!(sizes, k, children_arr[first(children_indices)])
            elseif op == :hcat
                total_cols = 0
                for c_idx in children_indices
                    total_cols +=
                        sizes.ndims[children_arr[c_idx]] <= 1 ? 1 :
                        _size(sizes, children_arr[c_idx], 2)
                end
                if sizes.ndims[children_arr[first(children_indices)]] == 0
                    shape = (1, total_cols)
                else
                    @assert sizes.ndims[children_arr[first(
                        children_indices,
                    )]] <= 2 "Hcat with ndims > 2 is not supported yet"
                    shape = (
                        _size(sizes, children_arr[first(children_indices)], 1),
                        total_cols,
                    )
                end
                _add_size!(sizes, k, tuple(shape...))
            elseif op == :vcat
                total_rows = 0
                for c_idx in children_indices
                    total_rows +=
                        sizes.ndims[children_arr[c_idx]] <= 1 ? 1 :
                        _size(sizes, children_arr[c_idx], 1)
                end
                if sizes.ndims[children_arr[first(children_indices)]] == 0
                    shape = (total_rows, 1)
                else
                    @assert sizes.ndims[children_arr[first(
                        children_indices,
                    )]] <= 2 "Hcat with ndims > 2 is not supported yet"
                    shape = (
                        total_rows,
                        _size(sizes, children_arr[first(children_indices)], 2),
                    )
                end
                _add_size!(sizes, k, tuple(shape...))
            elseif op == :*
                sizes.ndims[k] = 0
                for child in children_indices
                    id = children_arr[child]
                    ndims = sizes.ndims[id]
                    if !iszero(ndims)
                        sz = _size(sizes, id)
                        if iszero(sizes.ndims[k])
                            sizes.size_offset[k] = length(sizes.size)
                            append!(sizes.size, sz)
                            sizes.ndims[k] = ndims
                        else
                            @assert sizes.ndims[k] > 1
                            @assert sz[1] == sizes.size[end]
                            pop!(sizes.size)
                            append!(sizes.size, @view(sz[2:end]))
                            sizes.ndims[k] += ndims - 2
                        end
                    end
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
            elseif op == :sum_dims
                # Two args: (array, Vector{Float64}(dims)). Output keeps the
                # input ndims with the reduced dims collapsed to size 1.
                @assert N == 2 "`sum_dims` expects (array, dims_vector)"
                arr_id = children_arr[first(children_indices)]
                dims_id = children_arr[first(children_indices)+1]
                @assert nodes[dims_id].type == NODE_VALUE_BLOCK "`sum_dims` requires constant dims (NODE_VALUE_BLOCK)"
                # Read the dims values out of `const_values`. The block was
                # appended at `nodes[dims_id].index` with length recorded in
                # `block_shapes`.
                dims_len = prod(block_shapes[dims_id])
                start = nodes[dims_id].index
                dims_vec = const_values[(start-1) .+ (1:dims_len)]
                in_ndims = sizes.ndims[arr_id]
                out_shape = map(1:in_ndims) do d
                    return d in dims_vec ? 1 : _size(sizes, arr_id, d)
                end
                _add_size!(sizes, k, out_shape)
            else
                _assert_scalar_children(
                    sizes,
                    children_arr,
                    children_indices,
                    op,
                )
            end
        elseif node.type == NODE_CALL_MULTIVARIATE_BROADCASTED
            if !(node.index in eachindex(DEFAULT_MULTIVARIATE_OPERATORS))
                # TODO user-defined operators
                continue
            end
            op = DEFAULT_MULTIVARIATE_OPERATORS[node.index]
            if op == :+ || op == :- || op == :* || op == :/
                sizes.ndims[k] = maximum(children_indices, init = 0) do i
                    return sizes.ndims[children_arr[i]]
                end
                sizes.size_offset[k] = length(sizes.size)
                for _ in 1:sizes.ndims[k]
                    push!(sizes.size, 1)
                end
                sz_parent = _size(sizes, k)
                for i in children_indices
                    id = children_arr[i]
                    sz = _size(sizes, id)
                    for j in eachindex(sz)
                        if sz[j] > 1
                            if sz_parent[j] == 1
                                sz_parent[j] = sz[j]
                            else
                                @assert sz_parent[j] == sz[j]
                            end
                        end
                    end
                end
            elseif op == :^
                # Broadcasted ^ with scalar exponent preserves base shape
                @assert length(children_indices) == 2 "Expected two arguments for broadcasted operator `$op`, got $(length(children_indices))"
                @assert iszero(sizes.ndims[children_arr[children_indices[2]]]) "Expected scalar exponent for broadcasted operator `$op`"
                _copy_size!(sizes, k, children_arr[first(children_indices)])
            end
        elseif node.type == NODE_CALL_UNIVARIATE
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
        elseif node.type == NODE_CALL_UNIVARIATE_BROADCASTED
            if !(
                node.index in
                eachindex(MOI.Nonlinear.DEFAULT_UNIVARIATE_OPERATORS)
            )
                if operators !== nothing &&
                   node.index in eachindex(operators.univariate_operators) &&
                   haskey(
                       operators.chainrules_operators,
                       operators.univariate_operators[node.index],
                   )
                    @assert N == 1
                    _copy_size!(sizes, k, children_arr[first(children_indices)])
                    continue
                end
                error("TODO user-defined operators")
                continue
            end
            @assert N == 1
            op = MOI.Nonlinear.DEFAULT_UNIVARIATE_OPERATORS[node.index]
            _copy_size!(sizes, k, children_arr[first(children_indices)])
        end
    end
    for k in eachindex(nodes)
        sizes.storage_offset[k+1] = sizes.storage_offset[k] + _length(sizes, k)
    end
    return sizes
end

struct _SubexpressionStorage{T<:Real,S<:AbstractVector{T}}
    nodes::Vector{Node}
    adj::SparseArrays.SparseMatrixCSC{Bool,Int}
    sizes::Sizes
    const_values::Vector{T}
    forward_storage::S
    partials_storage::S
    reverse_storage::S
    partials_storage_ϵ::Vector{Float64}
    linearity::Linearity
    # ChainRules pullbacks captured during the forward pass, keyed by node
    # index. Lazily populated by `_forward_eval` for nodes whose operator was
    # registered via `register_chainrules_operator`. Read back by
    # `_reverse_eval` to propagate cotangents to the children.
    chainrules_pullbacks::Dict{Int,Any}

    function _SubexpressionStorage(
        nodes::Vector{Node},
        adj::SparseArrays.SparseMatrixCSC{Bool,Int},
        const_values::Vector{T},
        block_shapes::Dict{Int,Vector{Int}},
        partials_storage_ϵ::Vector{Float64},
        linearity::Linearity,
        operators = nothing,
        ::Type{S} = Vector{T},
    ) where {T<:Real,S<:AbstractVector{T}}
        sizes = _infer_sizes(nodes, adj, block_shapes, const_values, operators)
        N = _length(sizes)
        # Pre-load value blocks into forward_storage once at construction;
        # each block is a contiguous-to-contiguous bulk copy. Individual
        # `NODE_VALUE` scalars (rare — exponents, constant divisors, etc) and
        # variable nodes are loaded by `_forward_eval` in the per-node loop.
        cpu_buffer = zeros(T, N)
        for k in 1:length(nodes)
            node = nodes[k]
            if node.type == NODE_VALUE_BLOCK
                j = sizes.storage_offset[k] + 1
                len = _length(sizes, k)
                cpu_buffer[j:(j+len-1)] .=
                    view(const_values, node.index:(node.index+len-1))
            end
        end
        forward_storage = convert(S, cpu_buffer)
        return new{T,S}(
            nodes,
            adj,
            sizes,
            const_values,
            forward_storage,
            fill!(S(undef, N), zero(T)),  # partials_storage,
            fill!(S(undef, N), zero(T)),  # reverse_storage,
            partials_storage_ϵ,
            linearity,
            Dict{Int,Any}(),
        )
    end
end
