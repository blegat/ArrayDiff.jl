# Largely inspired by MathOptInterface/src/Nonlinear/model.jl
# Most functions have been copy-pasted and slightly modified to adapt to small changes in OperatorRegistry and Model.

function set_objective(model::Model, obj)
    model.objective = parse_expression(model, obj)
    return
end

function add_constraint(
    model::Model,
    func,
    set::Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.Interval{Float64},
        MOI.EqualTo{Float64},
    },
)
    f = parse_expression(model, func)
    model.last_constraint_index += 1
    index = MOI.Nonlinear.ConstraintIndex(model.last_constraint_index)
    model.constraints[index] = MOI.Nonlinear.Constraint(f, set)
    return index
end

function add_parameter(model::Model, value::Float64)
    push!(model.parameters, value)
    return MOI.Nonlinear.ParameterIndex(length(model.parameters))
end

function add_expression(model::Model, expr)
    push!(model.expressions, parse_expression(model, expr))
    return Nonlinear.ExpressionIndex(length(model.expressions))
end

function Base.getindex(model::Model, index::Nonlinear.ExpressionIndex)
    return model.expressions[index.value]
end

function register_operator(model::Model, op::Symbol, nargs::Int, f::Function...)
    return register_operator(model.operators, op, nargs, f...)
end

function register_operator(
    registry::OperatorRegistry,
    op::Symbol,
    nargs::Int,
    f::Function...,
)
    if nargs == 1
        if haskey(registry.univariate_operator_to_id, op)
            error("Operator $op is already registered.")
        elseif haskey(registry.multivariate_operator_to_id, op)
            error("Operator $op is already registered.")
        end
        operator = Nonlinear._UnivariateOperator(op, f...)
        push!(registry.univariate_operators, op)
        push!(registry.registered_univariate_operators, operator)
        registry.univariate_operator_to_id[op] =
            length(registry.univariate_operators)
    else
        if haskey(registry.multivariate_operator_to_id, op)
            error("Operator $op is already registered.")
        elseif haskey(registry.univariate_operator_to_id, op)
            error("Operator $op is already registered.")
        end
        operator = Nonlinear._MultivariateOperator{nargs}(op, f...)
        push!(registry.multivariate_operators, op)
        push!(registry.registered_multivariate_operators, operator)
        registry.multivariate_operator_to_id[op] =
            length(registry.multivariate_operators)
    end
    return
end

function features_available(evaluator::Evaluator)
    features = Symbol[]
    if evaluator.backend !== nothing
        append!(features, MOI.features_available(evaluator.backend))
    end
    if !(:ExprGraph in features)
        push!(features, :ExprGraph)
    end
    return features
end