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
    infer_sizes(op, child_sizes...) -> shape

Return the output shape of applying `op` to arguments of shapes `child_sizes`.
Each `child_sizes[i]` is empty if argument `i` is a scalar (`()` or `Int[]`),
or any indexable container of positive integers if it is an array. The
returned shape is empty for a scalar output.

Inputs may be tuples (JuMP-side, from `size()`) or `AbstractVector{Int}`
(tape-side — typically views into a `Sizes` buffer). The returned shape can
likewise be a tuple or an `AbstractVector{Int}` (including a view): pick
whichever makes the method type-stable. `_infer_sizes` and
`_build_user_op_expr` accept either.

The default implementation constructs dummy arguments with `zeros(sz)`
(or `0.0` for scalars) and calls `op(args...)`. Specialise on `op`'s
`typeof` to avoid the allocation, to support operators that error on zero
inputs, or to compute the output shape symbolically.

For tape-internal use, built-in operator symbols dispatch via
`infer_sizes(::Val{op}, shapes...)`, and broadcasted variants dispatch via
`infer_sizes(::Broadcasted{op}, shapes...)`.

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
function infer_sizes(op, child_sizes...)
    args = map(child_sizes) do sz
        return isempty(sz) ? 0.0 : zeros(sz...)
    end
    y = op(args...)
    return y isa AbstractArray ? size(y) : ()
end

# Symbol → Val redirect: built-in tape operators carry a `Symbol`, so callers
# from `_infer_sizes` invoke `infer_sizes(:op, shapes...)` which lands here.
infer_sizes(op::Symbol, shapes...) = infer_sizes(Val(op), shapes...)

"""
    Broadcasted{op}

Marker used to dispatch `infer_sizes` for the broadcasted variant of a
tape operator. Mirrors `Val(op)` but selects shape-combination rules that
follow Julia's broadcasting axis-combination rules instead of the
non-broadcasted (matmul/identity/etc.) rules.
"""
struct Broadcasted{S} end
Broadcasted(s::Symbol) = Broadcasted{s}()

# ── Built-in tape operators: non-broadcasted ────────────────────────────────
# These methods are written generically over indexable shape containers
# (tuple or `AbstractVector`), so the same definition serves both the
# JuMP-side (tuples from `size()`) and the tape-side (views into `Sizes`).

# Pure scalar reductions
infer_sizes(::Val{:sum}, shapes...) = ()
infer_sizes(::Val{:norm}, shapes...) = ()
infer_sizes(::Val{:dot}, shapes...) = ()

# vect: N scalar children → 1-D vector of length N
function infer_sizes(::Val{:vect}, shapes...)
    @assert all(isempty, shapes) "`vect` expects scalar children"
    return (length(shapes),)
end

# row: N scalar children → 1×N row vector
function infer_sizes(::Val{:row}, shapes...)
    @assert all(isempty, shapes) "`row` expects scalar children"
    return (1, length(shapes))
end

# +, - : non-broadcasted; all children share the same shape, output = first
infer_sizes(::Val{:+}, shape, more...) = shape
infer_sizes(::Val{:-}, shape, more...) = shape

# hcat: rows from first arg, total cols summed across children
function infer_sizes(::Val{:hcat}, shapes...)
    total_cols = sum(s -> length(s) <= 1 ? 1 : s[2], shapes)
    if isempty(shapes[1])
        return (1, total_cols)
    end
    @assert length(shapes[1]) <= 2 "hcat with ndims > 2 is not supported yet"
    return (shapes[1][1], total_cols)
end

# vcat: cols from first arg, total rows summed across children
function infer_sizes(::Val{:vcat}, shapes...)
    total_rows = sum(s -> length(s) <= 1 ? 1 : s[1], shapes)
    if isempty(shapes[1])
        return (total_rows, 1)
    end
    @assert length(shapes[1]) <= 2 "vcat with ndims > 2 is not supported yet"
    return (total_rows, shapes[1][2])
end

# *: matmul-like inner-dim reduction; scalar children are ignored. Returns a
# `Vector{Int}` so the accumulator is type-stable across the loop (a tuple
# accumulator would change type each iteration as the length varies).
function infer_sizes(::Val{:*}, shapes...)
    out = Int[]
    for s in shapes
        if isempty(s)
            continue
        end
        if isempty(out)
            append!(out, s)
        else
            @assert length(out) > 1
            @assert s[1] == out[end]
            pop!(out)
            for j in 2:length(s)
                push!(out, s[j])
            end
        end
    end
    return out
end

# ^ and / (non-broadcasted): first arg's shape, second must be scalar
function infer_sizes(::Val{:^}, base, exp)
    @assert isempty(exp) "`^` expects scalar exponent"
    return base
end
function infer_sizes(::Val{:/}, num, den)
    @assert isempty(den) "`/` expects scalar denominator"
    return num
end

# ── Built-in tape operators: broadcasted ────────────────────────────────────

# Broadcasted +, -, *, /: combine via Julia's broadcasting axis-combination
# rules. Output ndims = max child ndims; size in each dim is the max size > 1.
function infer_sizes(::Broadcasted{op}, shapes...) where {op}
    if !(op in (:+, :-, :*, :/))
        error("Unsupported broadcasted op `$op`")
    end
    nd = maximum(length, shapes; init = 0)
    out = ones(Int, nd)
    for sz in shapes
        for j in eachindex(sz)
            if sz[j] > 1
                if out[j] == 1
                    out[j] = sz[j]
                else
                    @assert out[j] == sz[j]
                end
            end
        end
    end
    return out
end

# Broadcasted ^: scalar exponent, base shape preserved
function infer_sizes(::Broadcasted{:^}, base, exp)
    @assert isempty(exp) "broadcasted `^` expects scalar exponent"
    return base
end

# `:sum_dims` is the one tape op whose shape depends on data outside the
# child-shape tuple (the constant dims vector), so it can't fit the generic
# `infer_sizes(op, shapes...)` signature. Handled inline by `_infer_sizes`.
function _sum_dims_shape(
    sizes,
    nodes,
    children_arr,
    children_indices,
    block_shapes,
    const_values,
)
    @assert length(children_indices) == 2 "`sum_dims` expects (array, dims_vector)"
    arr_id = children_arr[first(children_indices)]
    dims_id = children_arr[first(children_indices)+1]
    @assert nodes[dims_id].type == NODE_VALUE_BLOCK "`sum_dims` requires constant dims (NODE_VALUE_BLOCK)"
    dims_len = prod(block_shapes[dims_id])
    start = nodes[dims_id].index
    dims_vec = const_values[(start-1) .+ (1:dims_len)]
    in_ndims = sizes.ndims[arr_id]
    return map(1:in_ndims) do d
        return d in dims_vec ? 1 : _size(sizes, arr_id, d)
    end
end

# Resolve a tape node's "operator handle" — a `Symbol` (built-in op), a
# `Function` (user-defined ChainRules op), or `nothing` (unrecognised /
# scalar-only operator). For broadcasted multivariate nodes we wrap the
# symbol in `Broadcasted` so the corresponding `infer_sizes` method fires.
# This is type-unstable but it is only called from `_infer_sizes` for which
# performance isn't critical since it's just called at setup time.
# If performance is an issue, we can generate code with if-else like we
# already do at other places in this package.
function _shape_op(node::Node, operators)
    @assert node.type == NODE_CALL_MULTIVARIATE ||
            node.type == NODE_CALL_MULTIVARIATE_BROADCASTED
    func = nothing
    if node.index in eachindex(DEFAULT_MULTIVARIATE_OPERATORS)
        func = DEFAULT_MULTIVARIATE_OPERATORS[node.index]
    end
    if operators !== nothing &&
       node.index in eachindex(operators.multivariate_operators)
        op_sym = operators.multivariate_operators[node.index]
        if haskey(operators.chainrules_operators, op_sym)
            func = operators.chainrules_operators[op_sym]
        end
    end
    if isnothing(func)
        return nothing
    end
    if node.type == NODE_CALL_MULTIVARIATE_BROADCASTED
        func = Broadcasted(func)
    end
    return func
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
            _add_size!(sizes, k, block_shapes[k])
            continue
        end
        children_indices = SparseArrays.nzrange(adj, k)
        N = length(children_indices)
        if node.type == NODE_CALL_MULTIVARIATE ||
           node.type == NODE_CALL_MULTIVARIATE_BROADCASTED
            op = _shape_op(node, operators)
            if op === nothing
                continue  # TODO user-defined operators
            end
            if op === :sum_dims
                out_sz = _sum_dims_shape(
                    sizes,
                    nodes,
                    children_arr,
                    children_indices,
                    block_shapes,
                    const_values,
                )
            else
                # Pass shape views directly — no Tuple/Vector conversion. The
                # `infer_sizes` methods are generic over indexable containers.
                child_shapes = ntuple(
                    i -> _size(sizes, children_arr[children_indices[i]]),
                    N,
                )
                out_sz = infer_sizes(op, child_shapes...)
            end
            _add_size!(sizes, k, out_sz)
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
            end
            @assert N == 1
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
