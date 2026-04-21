# parse_expression methods for MOI function types on ArrayDiff.Model.

# ── Shared iterative stack loop ──────────────────────────────────────────────

function _parse_moi_stack(data::Model, expr::Expression, root, parent_index::Int)
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
    x::Union{Float64,MOI.VariableIndex},
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
                push!(
                    expr.nodes,
                    Node(NODE_CALL_UNIVARIATE, id, parent_index),
                )
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
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::ArrayOfContiguousVariables{2},
    parent_index::Int,
)
    m, n = x.size
    # Build vcat(row(v11, v12, ...), row(v21, v22, ...), ...)
    vcat_id = data.operators.multivariate_operator_to_id[:vcat]
    row_id = data.operators.multivariate_operator_to_id[:row]
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, vcat_id, parent_index))
    vcat_idx = length(expr.nodes)
    # Push rows in reverse order for stack processing
    for i in m:-1:1
        push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, row_id, vcat_idx))
        row_idx = length(expr.nodes)
        for j in n:-1:1
            vi = MOI.VariableIndex(x.offset + (j - 1) * m + i)
            push!(stack, (row_idx, vi))
        end
    end
    return
end

function _parse_moi_stack!(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::ArrayOfContiguousVariables{1},
    parent_index::Int,
)
    m = x.size[1]
    vect_id = data.operators.multivariate_operator_to_id[:vect]
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, vect_id, parent_index))
    vect_idx = length(expr.nodes)
    for i in m:-1:1
        vi = MOI.VariableIndex(x.offset + i)
        push!(stack, (vect_idx, vi))
    end
    return
end

# ── Constant matrices and vectors ────────────────────────────────────────────

function _parse_moi_stack!(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Matrix{Float64},
    parent_index::Int,
)
    m, n = size(x)
    vcat_id = data.operators.multivariate_operator_to_id[:vcat]
    row_id = data.operators.multivariate_operator_to_id[:row]
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, vcat_id, parent_index))
    vcat_idx = length(expr.nodes)
    for i in m:-1:1
        push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, row_id, vcat_idx))
        row_idx = length(expr.nodes)
        for j in n:-1:1
            push!(stack, (row_idx, x[i, j]))
        end
    end
    return
end

function _parse_moi_stack!(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Vector{Float64},
    parent_index::Int,
)
    vect_id = data.operators.multivariate_operator_to_id[:vect]
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, vect_id, parent_index))
    vect_idx = length(expr.nodes)
    for i in length(x):-1:1
        push!(stack, (vect_idx, x[i]))
    end
    return
end
