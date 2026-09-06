# Largely inspired by MathOptInterface/src/Nonlinear/model.jl
# Most functions have been copy-pasted and slightly modified to adapt to small changes in OperatorRegistry and Model.

function set_objective(model::Model, obj)
    model.objective = parse_expression(model, obj)
    if model.objective_sense == MOI.FEASIBILITY_SENSE
        model.objective_sense = MOI.MIN_SENSE
    end
    return
end

function set_residual!(model::Model, residual)
    model.residual = parse_expression(model, residual)
    return
end

function add_constraint(
    model::Model{T},
    func,
    set::Union{
        MOI.GreaterThan{T},
        MOI.LessThan{T},
        MOI.Interval{T},
        MOI.EqualTo{T},
    },
) where {T}
    f = parse_expression(model, func)
    model.last_constraint_index += 1
    index = ConstraintIndex(model.last_constraint_index)
    model.constraints[index] = Constraint(f, set)
    return index
end

function add_parameter(model::Model{T}, value::Real) where {T}
    push!(model.parameters, convert(T, value))
    return ParameterIndex(length(model.parameters))
end

function add_expression(model::Model, expr)
    push!(model.expressions, parse_expression(model, expr))
    return ExpressionIndex(length(model.expressions))
end

function Base.getindex(model::Model, index::ExpressionIndex)
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
        operator = _UnivariateOperator(op, f...)
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

"""
    register_chainrules_operator(model::OperatorRegistry, op::Symbol, f; arity::Int)

Register a user-defined array operator named `op` whose value is computed by
calling `f` and whose reverse-mode derivative is obtained through
`ChainRulesCore.rrule(f, args...)`.

`arity` is the number of arguments `f` takes. When `arity == 1`, the operator
is added to the univariate registry so it can be used in broadcasted form
(e.g. `relu.(x)`). When `arity > 1`, it is added to the multivariate registry
and can be applied to whole arrays (e.g. `crossentropy(p, q)`).
"""
function register_chainrules_operator(
    registry::OperatorRegistry,
    op::Symbol,
    f::Function;
    arity::Int,
)
    if haskey(registry.chainrules_operators, op)
        error("Chain-rules operator $op is already registered.")
    end
    if arity == 1
        if haskey(registry.univariate_operator_to_id, op)
            error("Operator $op is already registered.")
        end
        push!(registry.univariate_operators, op)
        registry.univariate_operator_to_id[op] =
            length(registry.univariate_operators)
    else
        if haskey(registry.multivariate_operator_to_id, op)
            error("Operator $op is already registered.")
        end
        push!(registry.multivariate_operators, op)
        registry.multivariate_operator_to_id[op] =
            length(registry.multivariate_operators)
    end
    registry.chainrules_operators[op] = f
    return
end

"""
    UserDefinedArrayOperator(name::Symbol; arity::Int) <: MOI.AbstractModelAttribute

Model-level attribute analogous to [`MOI.UserDefinedFunction`](@ref) used to
register a user-defined array operator whose reverse-mode derivative comes from
`ChainRulesCore.rrule`. Set it with the Julia function as the value, for
example `MOI.set(model, ArrayDiff.UserDefinedArrayOperator(:relu; arity = 1), relu)`.
"""
struct UserDefinedArrayOperator <: MOI.AbstractModelAttribute
    name::Symbol
    arity::Int
    UserDefinedArrayOperator(name::Symbol; arity::Int) = new(name, arity)
end

function MOI.set(model::Model, attr::UserDefinedArrayOperator, f::Function)
    register_chainrules_operator(
        model.operators,
        attr.name,
        f;
        arity = attr.arity,
    )
    return
end

MOI.supports(::Model, ::UserDefinedArrayOperator) = true

function MOI.features_available(evaluator::Evaluator)
    features = Symbol[]
    if evaluator.backend !== nothing
        append!(features, MOI.features_available(evaluator.backend))
    end
    if !(:ExprGraph in features)
        push!(features, :ExprGraph)
    end
    return features
end
