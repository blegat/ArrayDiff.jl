# Inspired by MathOptInterface/src/Nonlinear/parse_expression.jl

const DEFAULT_MULTIVARIATE_OPERATORS = [
    :+,
    :-,
    :*,
    :^,
    :/,
    :ifelse,
    :atan,
    :min,
    :max,
    :vect,
    :dot,
    :hcat,
    :vcat,
    :norm,
    :sum,
    :row,
]

struct OperatorRegistry
    # NODE_CALL_UNIVARIATE
    univariate_operators::Vector{Symbol}
    univariate_operator_to_id::Dict{Symbol,Int}
    univariate_user_operator_start::Int
    registered_univariate_operators::Vector{MOI.Nonlinear._UnivariateOperator}
    # NODE_CALL_MULTIVARIATE
    multivariate_operators::Vector{Symbol}
    multivariate_operator_to_id::Dict{Symbol,Int}
    multivariate_user_operator_start::Int
    registered_multivariate_operators::Vector{
        MOI.Nonlinear._MultivariateOperator,
    }
    # NODE_LOGIC
    logic_operators::Vector{Symbol}
    logic_operator_to_id::Dict{Symbol,Int}
    # NODE_COMPARISON
    comparison_operators::Vector{Symbol}
    comparison_operator_to_id::Dict{Symbol,Int}
    function OperatorRegistry()
        univariate_operators = copy(DEFAULT_UNIVARIATE_OPERATORS)
        multivariate_operators = copy(DEFAULT_MULTIVARIATE_OPERATORS)
        logic_operators = [:&&, :||]
        comparison_operators = [:<=, :(==), :>=, :<, :>]
        return new(
            # NODE_CALL_UNIVARIATE
            univariate_operators,
            Dict{Symbol,Int}(
                op => i for (i, op) in enumerate(univariate_operators)
            ),
            length(univariate_operators),
            _UnivariateOperator[],
            # NODE_CALL
            multivariate_operators,
            Dict{Symbol,Int}(
                op => i for (i, op) in enumerate(multivariate_operators)
            ),
            length(multivariate_operators),
            _MultivariateOperator[],
            # NODE_LOGIC
            logic_operators,
            Dict{Symbol,Int}(op => i for (i, op) in enumerate(logic_operators)),
            # NODE_COMPARISON
            comparison_operators,
            Dict{Symbol,Int}(
                op => i for (i, op) in enumerate(comparison_operators)
            ),
        )
    end
end

"""
    Model()

The core datastructure for representing a nonlinear optimization problem.

It has the following fields:
 * `objective::Union{Nothing,Expression}` : holds the nonlinear objective
   function, if one exists, otherwise `nothing`.
 * `expressions::Vector{Expression}` : a vector of expressions in the model.
 * `constraints::OrderedDict{ConstraintIndex,Constraint}` : a map from
   [`ConstraintIndex`](@ref) to the corresponding [`Constraint`](@ref). An
   `OrderedDict` is used instead of a `Vector` to support constraint deletion.
 * `parameters::Vector{Float64}` : holds the current values of the parameters.
 * `operators::OperatorRegistry` : stores the operators used in the model.
"""
mutable struct Model
    objective::Union{Nothing,MOI.Nonlinear.Expression}
    expressions::Vector{MOI.Nonlinear.Expression}
    constraints::OrderedDict{
        MOI.Nonlinear.ConstraintIndex,
        MOI.Nonlinear.Constraint,
    }
    parameters::Vector{Float64}
    operators::OperatorRegistry
    # This is a private field, used only to increment the ConstraintIndex.
    last_constraint_index::Int64
    function Model()
        model = MOI.Nonlinear.Model()
        ops = [:vect, :dot, :hcat, :vcat, :norm, :sum, :row]
        start = length(model.operators.multivariate_operators)
        append!(model.operators.multivariate_operators, ops)
        for (i, op) in enumerate(ops)
            model.operators.multivariate_operator_to_id[op] = start + i
        end
        return model
    end
end

function set_objective(model::MOI.Nonlinear.Model, obj)
    model.objective = parse_expression(model, obj)
    return
end

function parse_expression(data::MOI.Nonlinear.Model, input)
    expr = MOI.Nonlinear.Expression()
    parse_expression(data, expr, input, -1)
    return expr
end

function parse_expression(data, expr, item, parent)
    return MOI.Nonlinear.parse_expression(data, expr, item, parent)
end

function parse_expression(
    data::MOI.Nonlinear.Model,
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

function _parse_expression(stack, data, expr, x, parent_index)
    if Meta.isexpr(x, :call)
        if length(x.args) == 2 && !Meta.isexpr(x.args[2], :...)
            MOI.Nonlinear._parse_univariate_expression(
                stack,
                data,
                expr,
                x,
                parent_index,
            )
        else
            # The call is either n-ary, or it is a splat, in which case we
            # cannot tell just yet whether the expression is unary or nary.
            # Punt to multivariate and try to recover later.
            MOI.Nonlinear._parse_multivariate_expression(
                stack,
                data,
                expr,
                x,
                parent_index,
            )
        end
    elseif Meta.isexpr(x, :comparison)
        MOI.Nonlinear._parse_comparison_expression(
            stack,
            data,
            expr,
            x,
            parent_index,
        )
    elseif Meta.isexpr(x, :...)
        MOI.Nonlinear._parse_splat_expression(
            stack,
            data,
            expr,
            x,
            parent_index,
        )
    elseif Meta.isexpr(x, :&&) || Meta.isexpr(x, :||)
        MOI.Nonlinear._parse_logic_expression(
            stack,
            data,
            expr,
            x,
            parent_index,
        )
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

function eval_multivariate_function(
    registry::MOI.Nonlinear.OperatorRegistry,
    op::Symbol,
    x::AbstractVector{T},
) where {T}
    if op == :+
        return sum(x; init = zero(T))
    elseif op == :-
        @assert length(x) == 2
        return x[1] - x[2]
    elseif op == :*
        return prod(x; init = one(T))
    elseif op == :^
        @assert length(x) == 2
        # Use _nan_pow here to avoid throwing an error in common situations like
        # (-1.0)^1.5.
        return _nan_pow(x[1], x[2])
    elseif op == :/
        @assert length(x) == 2
        return x[1] / x[2]
    elseif op == :ifelse
        @assert length(x) == 3
        return ifelse(Bool(x[1]), x[2], x[3])
    elseif op == :atan
        @assert length(x) == 2
        return atan(x[1], x[2])
    elseif op == :min
        return minimum(x)
    elseif op == :max
        return maximum(x)
    elseif op == :vect
        return x
    end
    id = registry.multivariate_operator_to_id[op]
    offset = id - registry.multivariate_user_operator_start
    operator = registry.registered_multivariate_operators[offset]
    @assert length(x) == operator.N
    ret = operator.f(x)
    MOI.Nonlinear.check_return_type(T, ret)
    return ret::T
end

function _parse_vect_expression(
    stack::Vector{Tuple{Int,Any}},
    data::MOI.Nonlinear.Model,
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
    data::MOI.Nonlinear.Model,
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
    data::MOI.Nonlinear.Model,
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
    data::MOI.Nonlinear.Model,
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
