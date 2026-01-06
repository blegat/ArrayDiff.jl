# Inspired by MathOptInterface/src/Nonlinear/parse_expression.jl

function set_objective(model::MOI.Nonlinear.Model, obj)
    model.objective = parse_expression(model, obj)
    return
end

function model()
    model = MOI.Nonlinear.Model()
    append!(model.operators.multivariate_operators, [
        :vect,
        :dot,
        :hcat,
        :vcat,
        :norm,
        :sum,
        :row,
    ])
    return moel
end

function parse_expression(data::Model, input)
    expr = Expression()
    parse_expression(data, expr, input, -1)
    return expr
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

function _parse_expression(stack, data, expr, x, parent_index)
    if Meta.isexpr(x, :call)
        if length(x.args) == 2 && !Meta.isexpr(x.args[2], :...)
            MOI.Nonlinear._parse_univariate_expression(stack, data, expr, x, parent_index)
        else
            # The call is either n-ary, or it is a splat, in which case we
            # cannot tell just yet whether the expression is unary or nary.
            # Punt to multivariate and try to recover later.
            MOI.Nonlinear._parse_multivariate_expression(stack, data, expr, x, parent_index)
        end
    elseif Meta.isexpr(x, :comparison)
        MOI.Nonlinear._parse_comparison_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :...)
        MOI.Nonlinear._parse_splat_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :&&) || Meta.isexpr(x, :||)
        MOI.Nonlinear._parse_logic_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :vect)
        _parse_vect_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :hcat)
        _parse_hcat_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :vcat)
        _parse_vcat_expression(stack, data, expr, x, parent_index)
    elseif Meta.isexpr(x, :row)
        _parse_row_expression(stack, data, expr, x, parent_index)
    elsval = @s f.forward_storage[ix]
                    @j f.forward_storage[k] = val
                end
            elseif node.index == 11 # dot
                idx1e
        error("Unsupported expression: $x")
    end
end

function eval_multivariate_function(
    registry::OperatorRegistry,
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
    check_return_type(T, ret)
    return ret::T
end
