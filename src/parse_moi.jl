# parse_expression methods for MOI function types on ArrayDiff.Model.

# ── Shared iterative stack loop ──────────────────────────────────────────────

function _parse_moi_stack(
    data::Model,
    expr::Expression,
    root,
    parent_index::Int,
)
    stack = Tuple{Int,Any}[(parent_index, root)]
    while !isempty(stack)
        parent, item = pop!(stack)
        _parse_moi_stack!(stack, data, expr, item, parent)
    end
    return
end

# ── Entry points ─────────────────────────────────────────────────────────────

function parse_expression(
    data::Model,
    expr::Expression,
    x::MOI.ScalarNonlinearFunction,
    parent_index::Int,
)
    return _parse_moi_stack(data, expr, x, parent_index)
end

function parse_expression(
    data::Model,
    expr::Expression,
    x::ArrayNonlinearFunction,
    parent_index::Int,
)
    return _parse_moi_stack(data, expr, x, parent_index)
end

function parse_expression(
    data::Model,
    expr::Expression,
    x::ArrayOfContiguousVariables,
    parent_index::Int,
)
    return _parse_moi_stack(data, expr, x, parent_index)
end

# ── ScalarNonlinearFunction ──────────────────────────────────────────────────

function _parse_moi_stack!(
    ::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Union{Real,MOI.VariableIndex},
    parent_index::Int,
)
    return parse_expression(data, expr, x, parent_index)
end

function _parse_moi_stack!(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::MOI.ScalarNonlinearFunction,
    parent_index::Int,
)
    op = x.head
    nargs = length(x.args)
    if nargs == 1
        id = get(data.operators.univariate_operator_to_id, op, nothing)
        if id !== nothing
            push!(expr.nodes, Node(NODE_CALL_UNIVARIATE, id, parent_index))
            push!(stack, (length(expr.nodes), x.args[1]))
            return
        end
    end
    id = get(data.operators.multivariate_operator_to_id, op, nothing)
    if id === nothing
        throw(MOI.UnsupportedNonlinearOperator(op))
    end
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, id, parent_index))
    for i in nargs:-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

# ── ArrayNonlinearFunction ───────────────────────────────────────────────────

function _parse_moi_stack!(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::ArrayNonlinearFunction,
    parent_index::Int,
)
    op = x.head
    nargs = length(x.args)
    if x.broadcasted
        if nargs == 1
            id = get(data.operators.univariate_operator_to_id, op, nothing)
            if id !== nothing
                push!(
                    expr.nodes,
                    Node(NODE_CALL_UNIVARIATE_BROADCASTED, id, parent_index),
                )
                push!(stack, (length(expr.nodes), x.args[1]))
                return
            end
        end
        id = get(data.operators.multivariate_operator_to_id, op, nothing)
        if id === nothing
            throw(MOI.UnsupportedNonlinearOperator(op))
        end
        push!(
            expr.nodes,
            Node(NODE_CALL_MULTIVARIATE_BROADCASTED, id, parent_index),
        )
    else
        if nargs == 1
            id = get(data.operators.univariate_operator_to_id, op, nothing)
            if id !== nothing
                push!(expr.nodes, Node(NODE_CALL_UNIVARIATE, id, parent_index))
                push!(stack, (length(expr.nodes), x.args[1]))
                return
            end
        end
        id = get(data.operators.multivariate_operator_to_id, op, nothing)
        if id === nothing
            throw(MOI.UnsupportedNonlinearOperator(op))
        end
        push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, id, parent_index))
    end
    for i in nargs:-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

# ── ArrayOfContiguousVariables ───────────────────────────────────────────────────

function _parse_moi_stack!(
    ::Vector{Tuple{Int,Any}},
    ::Model,
    expr::Expression,
    x::ArrayOfContiguousVariables,
    parent_index::Int,
)
    # Emit a single block node. The block represents the contiguous range of
    # MOI variable indices `x.offset+1, ...`, laid out in
    # column-major order (matching `Array{Float64}` and `Base.LinearIndices`),
    # which is the layout `_view_array` will see at evaluation time.
    push!(expr.nodes, Node(NODE_MOI_VARIABLE_BLOCK, x.offset + 1, parent_index))
    expr.block_shapes[length(expr.nodes)] = collect(x.size)
    return
end

# ── Constant arrays ────────────────────────────────────────────

function _parse_moi_stack!(
    ::Vector{Tuple{Int,Any}},
    ::Model,
    expr::Expression,
    x::AbstractArray{<:Real},
    parent_index::Int,
)
    # Emit a single value block. We push the flat values to
    # `expr.values` in column-major order (matching `Array{Float64}`'s memory
    # layout); `node.index` records the start of that contiguous range so
    # `_SubexpressionStorage` can copy it into the tape in one block at
    # construction time.
    start_idx = length(expr.values) + 1
    append!(expr.values, x)
    push!(expr.nodes, Node(NODE_VALUE_BLOCK, start_idx, parent_index))
    expr.block_shapes[length(expr.nodes)] = collect(size(x))
    return
end
