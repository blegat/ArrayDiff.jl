# Largely inspired by MathOptInterface/src/Nonlinear/parse.jl
# Most functions have been copy-pasted and slightly modified to adapt to small changes in OperatorRegistry and Model.

function _parse_multivariate_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :call)
    broadcasted = false
    # if first char of x is a dot, then it is broadcasted and we should look up the operator without the dot
    if x.args[1] isa Symbol && startswith(string(x.args[1]), ".")
        x = Expr(:call, Symbol(string(x.args[1])[2:end]), x.args[2:end]...)
        broadcasted = true
    end
    id = get(data.operators.multivariate_operator_to_id, x.args[1], nothing)
    if id === nothing
        if haskey(data.operators.univariate_operator_to_id, x.args[1])
            # It may also be a unary variate operator with splatting.
            _parse_univariate_expression(stack, data, expr, x, parent_index)
        elseif x.args[1] in data.operators.comparison_operators
            # Or it may be a binary (in)equality operator.
            _parse_inequality_expression(stack, data, expr, x, parent_index)
        else
            throw(MOI.UnsupportedNonlinearOperator(x.args[1]))
        end
        return
    end
    if broadcasted
        push!(
            expr.nodes,
            Node(NODE_CALL_MULTIVARIATE_BROADCASTED, id, parent_index),
        )
    else
        push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, id, parent_index))
    end
    for i in length(x.args):-1:2
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function parse_expression(
    ::Model,
    expr::Expression,
    x::MOI.VariableIndex,
    parent_index::Int,
)
    push!(expr.nodes, Node(NODE_MOI_VARIABLE, x.value, parent_index))
    return
end

function parse_expression(data::Model, input)
    expr = Expression()
    parse_expression(data, expr, input, -1)
    return expr
end

function parse_expression(::Model, expr::Expression, x::Real, parent_index::Int)
    push!(expr.values, convert(Float64, x)::Float64)
    push!(expr.nodes, Node(NODE_VALUE, length(expr.values), parent_index))
    return
end

function parse_expression(
    ::Model,
    expr::Expression,
    x::ParameterIndex,
    parent_index::Int,
)
    push!(expr.nodes, Node(NODE_PARAMETER, x.value, parent_index))
    return
end

function parse_expression(
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    stack = Tuple{Int,Any}[]
    push!(stack, (parent_index, x))
    while !isempty(stack)
        parent, item = pop!(stack)
        if item isa Expr
            _parse_expression(stack, data, expr, item, parent)
        else
            parse_expression(data, expr, item, parent)
        end
    end
    return
end

function parse_expression(
    ::Model,
    expr::Expression,
    x::ExpressionIndex,
    parent_index::Int,
)
    push!(expr.nodes, Node(NODE_SUBEXPRESSION, x.value, parent_index))
    return
end

function _parse_univariate_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :call, 2) || Meta.isexpr(x, :., 2)
    broadcasted = false
    if Meta.isexpr(x, :.)
        broadcasted = true
    end
    # if first char of x is a dot, then it is broadcasted and we should look up the operator without the dot
    if x.args[1] isa Symbol && startswith(string(x.args[1]), ".")
        x = Expr(:call, Symbol(string(x.args[1])[2:end]), x.args[2:end]...)
        broadcasted = true
    end
    id = get(data.operators.univariate_operator_to_id, x.args[1], nothing)
    if id === nothing
        # It may also be a multivariate operator like * with one argument.
        if haskey(data.operators.multivariate_operator_to_id, x.args[1])
            _parse_multivariate_expression(stack, data, expr, x, parent_index)
            return
        end
        throw(MOI.UnsupportedNonlinearOperator(x.args[1]))
    end
    if broadcasted
        push!(
            expr.nodes,
            Node(NODE_CALL_UNIVARIATE_BROADCASTED, id, parent_index),
        )
    else
        push!(expr.nodes, Node(NODE_CALL_UNIVARIATE, id, parent_index))
    end
    push!(stack, (length(expr.nodes), x.args[2]))
    return
end

function _parse_logic_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    id = data.operators.logic_operator_to_id[x.head]
    push!(expr.nodes, Node(NODE_LOGIC, id, parent_index))
    parent_var = length(expr.nodes)
    push!(stack, (parent_var, x.args[2]))
    push!(stack, (parent_var, x.args[1]))
    return
end

function _parse_expression(stack, data, expr, x, parent_index)
    if Meta.isexpr(x, :call)
        if x.args[1] == :reduce
            _parse_reduce_expression(stack, data, expr, x, parent_index)
        elseif length(x.args) == 2 && !Meta.isexpr(x.args[2], :...)
            _parse_univariate_expression(stack, data, expr, x, parent_index)
        else
            # The call is either n-ary, or it is a splat, in which case we
            # cannot tell just yet whether the expression is unary or nary.
            # Punt to multivariate and try to recover later.
            _parse_multivariate_expression(stack, data, expr, x, parent_index)
        end
    elseif Meta.isexpr(x, :.)
        # This is a special case for handling univariate broadcasted operators
        _parse_univariate_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :comparison)
        _parse_comparison_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :...)
        MOI.Nonlinear._parse_splat_expression(
            stack,
            data,
            expr,
            x,
            parent_index,
        )
    elseif Meta.isexpr(x, :&&) || Meta.isexpr(x, :||)
        _parse_logic_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :vect)
        _parse_vect_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :hcat)
        _parse_hcat_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :vcat)
        _parse_vcat_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :row)
        _parse_row_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :tuple) && length(x.args) == 1
        push!(stack, (parent_index, x.args[1]))
    else
        error("Unsupported expression: $x")
    end
end

function _parse_comparison_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    for k in 2:2:(length(x.args)-1)
        @assert x.args[k] == x.args[2] # don't handle a <= b >= c
    end
    operator_id = data.operators.comparison_operator_to_id[x.args[2]]
    push!(expr.nodes, Node(NODE_COMPARISON, operator_id, parent_index))
    for i in length(x.args):-2:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_vect_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :vect)
    id = get(data.operators.multivariate_operator_to_id, :vect, nothing)
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, id, parent_index))
    for i in length(x.args):-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_row_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :row)
    id = get(data.operators.multivariate_operator_to_id, :row, nothing)
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, id, parent_index))
    for i in length(x.args):-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_hcat_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :hcat)
    id = get(data.operators.multivariate_operator_to_id, :hcat, nothing)
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, id, parent_index))
    for i in length(x.args):-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_vcat_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :vcat)
    id = get(data.operators.multivariate_operator_to_id, :vcat, nothing)
    push!(expr.nodes, Node(NODE_CALL_MULTIVARIATE, id, parent_index))
    for i in length(x.args):-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_reduce_expression(stack, data, expr, x, parent_index)
    if length(x.args) != 3
        error(
            "Unsupported reduce expression: $x. Expected reduce(op, collection).",
        )
    end

    op = x.args[2]
    collection = x.args[3]

    if !Meta.isexpr(collection, :vect)
        error(
            "Unsupported reduce collection: $collection. Expected a vector literal.",
        )
    end

    args = collection.args

    if isempty(args)
        error("Unsupported reduce on empty collection.")
    elseif length(args) == 1
        push!(stack, (parent_index, args[1]))
        return
    end

    folded = Expr(:call, op, args[1], args[2])
    for i in 3:length(args)
        folded = Expr(:call, op, folded, args[i])
    end

    push!(stack, (parent_index, folded))
    return
end

function _parse_inequality_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Expression,
    x::Expr,
    parent_index::Int,
)
    operator_id = data.operators.comparison_operator_to_id[x.args[1]]
    push!(expr.nodes, Node(NODE_COMPARISON, operator_id, parent_index))
    for i in length(x.args):-1:2
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end
