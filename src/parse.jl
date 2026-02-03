# Largely inspired by MathOptInterface/src/Nonlinear/parse.jl
# Most functions have been copy-pasted and slightly modified to adapt to small changes in OperatorRegistry and Model.

function _parse_multivariate_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::MOI.Nonlinear.Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :call)
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
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(
            MOI.Nonlinear.NODE_CALL_MULTIVARIATE,
            id,
            parent_index,
        ),
    )
    for i in length(x.args):-1:2
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function parse_expression(
    ::Model,
    expr::MOI.Nonlinear.Expression,
    x::MOI.VariableIndex,
    parent_index::Int,
)
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(
            MOI.Nonlinear.NODE_MOI_VARIABLE,
            x.value,
            parent_index,
        ),
    )
    return
end

function parse_expression(data::Model, input)
    expr = MOI.Nonlinear.Expression()
    parse_expression(data, expr, input, -1)
    return expr
end

function parse_expression(
    ::Model,
    expr::MOI.Nonlinear.Expression,
    x::Real,
    parent_index::Int,
)
    push!(expr.values, convert(Float64, x)::Float64)
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(
            MOI.Nonlinear.NODE_VALUE,
            length(expr.values),
            parent_index,
        ),
    )
    return
end

function parse_expression(
    ::Model,
    expr::MOI.Nonlinear.Expression,
    x::MOI.Nonlinear.ParameterIndex,
    parent_index::Int,
)
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(MOI.Nonlinear.NODE_PARAMETER, x.value, parent_index),
    )
    return
end

function parse_expression(
    data::Model,
    expr::MOI.Nonlinear.Expression,
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
    expr::Nonlinear.Expression,
    x::Nonlinear.ExpressionIndex,
    parent_index::Int,
)
    push!(
        expr.nodes,
        Nonlinear.Node(Nonlinear.NODE_SUBEXPRESSION, x.value, parent_index),
    )
    return
end

function _parse_univariate_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::MOI.Nonlinear.Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :call, 2)
    id = get(data.operators.univariate_operator_to_id, x.args[1], nothing)
    if id === nothing
        # It may also be a multivariate operator like * with one argument.
        if haskey(data.operators.multivariate_operator_to_id, x.args[1])
            _parse_multivariate_expression(stack, data, expr, x, parent_index)
            return
        end
        throw(MOI.UnsupportedNonlinearOperator(x.args[1]))
    end
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(
            MOI.Nonlinear.NODE_CALL_UNIVARIATE,
            id,
            parent_index,
        ),
    )
    push!(stack, (length(expr.nodes), x.args[2]))
    return
end

function _parse_logic_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::MOI.Nonlinear.Expression,
    x::Expr,
    parent_index::Int,
)
    id = data.operators.logic_operator_to_id[x.head]
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(MOI.Nonlinear.NODE_LOGIC, id, parent_index),
    )
    parent_var = length(expr.nodes)
    push!(stack, (parent_var, x.args[2]))
    push!(stack, (parent_var, x.args[1]))
    return
end

function _parse_expression(stack, data, expr, x, parent_index)
    if Meta.isexpr(x, :call)
        if length(x.args) == 2 && !Meta.isexpr(x.args[2], :...)
            _parse_univariate_expression(stack, data, expr, x, parent_index)
        else
            # The call is either n-ary, or it is a splat, in which case we
            # cannot tell just yet whether the expression is unary or nary.
            # Punt to multivariate and try to recover later.
            _parse_multivariate_expression(stack, data, expr, x, parent_index)
        end
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
    else
        error("Unsupported expression: $x")
    end
end

function _parse_comparison_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Nonlinear.Expression,
    x::Expr,
    parent_index::Int,
)
    for k in 2:2:(length(x.args)-1)
        @assert x.args[k] == x.args[2] # don't handle a <= b >= c
    end
    operator_id = data.operators.comparison_operator_to_id[x.args[2]]
    push!(
        expr.nodes,
        Nonlinear.Node(Nonlinear.NODE_COMPARISON, operator_id, parent_index),
    )
    for i in length(x.args):-2:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_vect_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::MOI.Nonlinear.Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :vect)
    id = get(data.operators.multivariate_operator_to_id, :vect, nothing)
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(
            MOI.Nonlinear.NODE_CALL_MULTIVARIATE,
            id,
            parent_index,
        ),
    )
    for i in length(x.args):-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_row_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::MOI.Nonlinear.Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :row)
    id = get(data.operators.multivariate_operator_to_id, :row, nothing)
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(
            MOI.Nonlinear.NODE_CALL_MULTIVARIATE,
            id,
            parent_index,
        ),
    )
    for i in length(x.args):-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_hcat_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::MOI.Nonlinear.Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :hcat)
    id = get(data.operators.multivariate_operator_to_id, :hcat, nothing)
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(
            MOI.Nonlinear.NODE_CALL_MULTIVARIATE,
            id,
            parent_index,
        ),
    )
    for i in length(x.args):-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_vcat_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::MOI.Nonlinear.Expression,
    x::Expr,
    parent_index::Int,
)
    @assert Meta.isexpr(x, :vcat)
    id = get(data.operators.multivariate_operator_to_id, :vcat, nothing)
    push!(
        expr.nodes,
        MOI.Nonlinear.Node(
            MOI.Nonlinear.NODE_CALL_MULTIVARIATE,
            id,
            parent_index,
        ),
    )
    for i in length(x.args):-1:1
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end

function _parse_inequality_expression(
    stack::Vector{Tuple{Int,Any}},
    data::Model,
    expr::Nonlinear.Expression,
    x::Expr,
    parent_index::Int,
)
    operator_id = data.operators.comparison_operator_to_id[x.args[1]]
    push!(
        expr.nodes,
        Nonlinear.Node(Nonlinear.NODE_COMPARISON, operator_id, parent_index),
    )
    for i in length(x.args):-1:2
        push!(stack, (length(expr.nodes), x.args[i]))
    end
    return
end