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
        univariate_operators = copy(MOI.Nonlinear.DEFAULT_UNIVARIATE_OPERATORS)
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
            MOI.Nonlinear._UnivariateOperator[],
            # NODE_CALL
            multivariate_operators,
            Dict{Symbol,Int}(
                op => i for (i, op) in enumerate(multivariate_operators)
            ),
            length(multivariate_operators),
            MOI.Nonlinear._MultivariateOperator[],
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
        model = new(
            nothing,
            MOI.Nonlinear.Expression[],
            OrderedDict{
                MOI.Nonlinear.ConstraintIndex,
                MOI.Nonlinear.Constraint,
            }(),
            Float64[],
            OperatorRegistry(),
            0,
        )
        return model
    end
end

_bound(s::MOI.LessThan) = MOI.NLPBoundsPair(-Inf, s.upper)
_bound(s::MOI.GreaterThan) = MOI.NLPBoundsPair(s.lower, Inf)
_bound(s::MOI.EqualTo) = MOI.NLPBoundsPair(s.value, s.value)
_bound(s::MOI.Interval) = MOI.NLPBoundsPair(s.lower, s.upper)

mutable struct Evaluator{B} <: MOI.AbstractNLPEvaluator
    # The internal datastructure.
    model::Model
    # The abstract-differentiation backend
    backend::B
    # ordered_constraints is needed because `OrderedDict` doesn't support
    # looking up a key by the linear index.
    ordered_constraints::Vector{MOI.Nonlinear.ConstraintIndex}
    # Storage for the NLPBlockDual, so that we can query the dual of individual
    # constraints without needing to query the full vector each time.
    constraint_dual::Vector{Float64}
    # Timers
    initialize_timer::Float64
    eval_objective_timer::Float64
    eval_constraint_timer::Float64
    eval_objective_gradient_timer::Float64
    eval_constraint_gradient_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_objective_timer::Float64
    eval_hessian_constraint_timer::Float64
    eval_hessian_lagrangian_timer::Float64

    function Evaluator(
        model::Model,
        backend::B = nothing,
    ) where {B<:Union{Nothing,MOI.AbstractNLPEvaluator}}
        return new{B}(
            model,
            backend,
            MOI.ConstraintIndex[],
            Float64[],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    end
end

"""
    MOI.NLPBlockData(evaluator::Evaluator)

Create an [`MOI.NLPBlockData`](@ref) object from an [`Evaluator`](@ref)
object.
"""
function MOI.NLPBlockData(evaluator::Evaluator)
    return MOI.NLPBlockData(
        [_bound(c.set) for (_, c) in evaluator.model.constraints],
        evaluator,
        evaluator.model.objective !== nothing,
    )
end

"""
    ExprGraphOnly() <: AbstractAutomaticDifferentiation

The default implementation of `AbstractAutomaticDifferentiation`. The only
supported feature is `:ExprGraph`.
"""
struct ExprGraphOnly <: MOI.Nonlinear.AbstractAutomaticDifferentiation end

function Evaluator(model::Model, ::ExprGraphOnly, ::Vector{MOI.VariableIndex})
    return Evaluator(model)
end

"""
    SparseReverseMode() <: AbstractAutomaticDifferentiation

An implementation of `AbstractAutomaticDifferentiation` that uses sparse
reverse-mode automatic differentiation to compute derivatives. Supports all
features in the MOI nonlinear interface.
"""
struct SparseReverseMode <: MOI.Nonlinear.AbstractAutomaticDifferentiation end

function Evaluator(
    model::Model,
    ::SparseReverseMode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    return Evaluator(model, ReverseAD.NLPEvaluator(model, ordered_variables))
end

"""
    SymbolicMode() <: AbstractAutomaticDifferentiation

A type for setting as the value of the `MOI.AutomaticDifferentiationBackend()`
attribute to enable symbolic automatic differentiation.
"""
struct SymbolicMode <: MOI.Nonlinear.AbstractAutomaticDifferentiation end

function Evaluator(
    model::Model,
    ::SymbolicMode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    return Evaluator(model, SymbolicAD.Evaluator(model, ordered_variables))
end

function set_objective(model::Model, obj)
    model.objective = parse_expression(model, obj)
    return
end

function set_objective(model::Model, ::Nothing)
    model.objective = nothing
    return
end

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

function eval_logic_function(
    ::OperatorRegistry,
    op::Symbol,
    lhs::T,
    rhs::T,
)::Bool where {T}
    if op == :&&
        return lhs && rhs
    else
        @assert op == :||
        return lhs || rhs
    end
end

function MOI.initialize(evaluator::Evaluator, features::Vector{Symbol})
    start_time = time()
    empty!(evaluator.ordered_constraints)
    evaluator.eval_objective_timer = 0.0
    evaluator.eval_objective_gradient_timer = 0.0
    evaluator.eval_constraint_timer = 0.0
    evaluator.eval_constraint_gradient_timer = 0.0
    evaluator.eval_constraint_jacobian_timer = 0.0
    evaluator.eval_hessian_objective_timer = 0.0
    evaluator.eval_hessian_constraint_timer = 0.0
    evaluator.eval_hessian_lagrangian_timer = 0.0
    append!(evaluator.ordered_constraints, keys(evaluator.model.constraints))
    # Every backend supports :ExprGraph, so don't forward it.
    filter!(f -> f != :ExprGraph, features)
    if evaluator.backend !== nothing
        MOI.initialize(evaluator.backend, features)
    elseif !isempty(features)
        @assert evaluator.backend === nothing  # ==> ExprGraphOnly used
        error(
            "Unable to initialize `Nonlinear.Evaluator` because the " *
            "following features are not supported: $features",
        )
    end
    evaluator.initialize_timer = time() - start_time
    return
end

function MOI.eval_objective(evaluator::Evaluator, x)
    start = time()
    obj = MOI.eval_objective(evaluator.backend, x)
    evaluator.eval_objective_timer += time() - start
    return obj
end

function MOI.eval_objective_gradient(evaluator::Evaluator, g, x)
    start = time()
    MOI.eval_objective_gradient(evaluator.backend, g, x)
    evaluator.eval_objective_gradient_timer += time() - start
    return
end

function MOI.hessian_lagrangian_structure(evaluator::Evaluator)
    return MOI.hessian_lagrangian_structure(evaluator.backend)
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
    MOI.Nonlinear.check_return_type(T, ret)
    return ret::T
end

function eval_multivariate_hessian(
    registry::OperatorRegistry,
    op::Symbol,
    H,
    x::AbstractVector{T},
) where {T}
    if op in (:+, :-, :ifelse)
        return false
    end
    if op == :*
        # f(x)    = *(x[i] for i in 1:N)
        #
        # ∇fᵢ(x)  = *(x[j] for j in 1:N if i != j)
        #
        # ∇fᵢⱼ(x) = *(x[k] for k in 1:N if i != k & j != k)
        N = length(x)
        if N == 1
            # Hessian is zero
        elseif N == 2
            H[2, 1] = one(T)
        else
            for i in 1:N, j in (i+1):N
                H[j, i] =
                    prod(x[k] for k in 1:N if k != i && k != j; init = one(T))
            end
        end
    elseif op == :^
        # f(x)   = x[1]^x[2]
        #
        # ∇f(x)  = x[2]*x[1]^(x[2]-1)
        #          x[1]^x[2]*log(x[1])
        #
        # ∇²f(x) = x[2]*(x[2]-1)*x[1]^(x[2]-2)
        #          x[1]^(x[2]-1)*(x[2]*log(x[1])+1) x[1]^x[2]*log(x[1])^2
        ln = x[1] > 0 ? log(x[1]) : NaN
        if x[2] == one(T)
            H[2, 1] = _nan_to_zero(ln + one(T))
            H[2, 2] = _nan_to_zero(x[1] * ln^2)
        elseif x[2] == T(2)
            H[1, 1] = T(2)
            H[2, 1] = _nan_to_zero(x[1] * (T(2) * ln + one(T)))
            H[2, 2] = _nan_to_zero(ln^2 * x[1]^2)
        else
            H[1, 1] = _nan_to_zero(x[2] * (x[2] - 1) * _nan_pow(x[1], x[2] - 2))
            H[2, 1] = _nan_to_zero(_nan_pow(x[1], x[2] - 1) * (x[2] * ln + 1))
            H[2, 2] = _nan_to_zero(ln^2 * _nan_pow(x[1], x[2]))
        end
    elseif op == :/
        # f(x)  = x[1]/x[2]
        #
        # ∇f(x) = 1/x[2]
        #         -x[1]/x[2]^2
        #
        # ∇²(x) = 0.0
        #         -1/x[2]^2 2x[1]/x[2]^3
        d = 1 / x[2]^2
        H[2, 1] = -d
        H[2, 2] = 2 * x[1] * d / x[2]
    elseif op == :atan
        # f(x)  = atan(y, x)
        #
        # ∇f(x) = +x/(x^2+y^2)
        #         -y/(x^2+y^2)
        #
        # ∇²(x) = -(2xy)/(x^2+y^2)^2
        #         (y^2-x^2)/(x^2+y^2)^2 (2xy)/(x^2+y^2)^2
        base = (x[1]^2 + x[2]^2)^2
        H[1, 1] = -2 * x[2] * x[1] / base
        H[2, 1] = (x[1]^2 - x[2]^2) / base
        H[2, 2] = 2 * x[2] * x[1] / base
    elseif op == :min
        _, i = findmin(x)
        H[i, i] = one(T)
    elseif op == :max
        _, i = findmax(x)
        H[i, i] = one(T)
    else
        id = registry.multivariate_operator_to_id[op]
        offset = id - registry.multivariate_user_operator_start
        operator = registry.registered_multivariate_operators[offset]
        if operator.∇²f === nothing
            error("Hessian is not defined for operator $op")
        end
        @assert length(x) == operator.N
        operator.∇²f(H, x)
    end
    return true
end

function eval_univariate_function_and_gradient(
    registry::OperatorRegistry,
    id::Integer,
    x::T,
) where {T}
    if id <= registry.univariate_user_operator_start
        return Nonlinear._eval_univariate(id, x)::Tuple{T,T}
    end
    offset = id - registry.univariate_user_operator_start
    operator = registry.registered_univariate_operators[offset]
    return Nonlinear.eval_univariate_function_and_gradient(operator, x)
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

function add_constraint(
    model::Model,
    v::Vector{MOI.VariableIndex},
    set::MOI.AbstractVectorSet,
)
    return add_constraint(model, VectorOfVariables(v), set)
end

"""
    add_constraints(model::ModelLike, funcs::Vector{F}, sets::Vector{S})::Vector{ConstraintIndex{F,S}} where {F,S}

Add the set of constraints specified by each function-set pair in `funcs` and `sets`. `F` and `S` should be concrete types.
This call is equivalent to `add_constraint.(model, funcs, sets)` but may be more efficient.
"""
function add_constraints end

# default fallback
function add_constraints(model::Model, funcs, sets)
    return add_constraint.(model, funcs, sets)
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

function MOI.eval_constraint_gradient(evaluator::Evaluator, ∇g, x, i)
    start = time()
    MOI.eval_constraint_gradient(evaluator.backend, ∇g, x, i)
    evaluator.eval_constraint_gradient_timer += time() - start
    return
end

function MOI.constraint_gradient_structure(evaluator::Evaluator, i)
    return MOI.constraint_gradient_structure(evaluator.backend, i)
end

function MOI.eval_constraint(evaluator::Evaluator, g, x)
    start = time()
    MOI.eval_constraint(evaluator.backend, g, x)
    evaluator.eval_constraint_timer += time() - start
    return
end

function MOI.jacobian_structure(evaluator::Evaluator)
    return MOI.jacobian_structure(evaluator.backend)
end

function MOI.eval_constraint_jacobian(evaluator::Evaluator, J, x)
    start = time()
    MOI.eval_constraint_jacobian(evaluator.backend, J, x)
    evaluator.eval_constraint_jacobian_timer += time() - start
    return
end

function MOI.hessian_objective_structure(evaluator::Evaluator)
    return MOI.hessian_objective_structure(evaluator.backend)
end

function MOI.hessian_constraint_structure(evaluator::Evaluator, i)
    return MOI.hessian_constraint_structure(evaluator.backend, i)
end

function MOI.hessian_lagrangian_structure(evaluator::Evaluator)
    return MOI.hessian_lagrangian_structure(evaluator.backend)
end

function MOI.eval_hessian_objective(evaluator::Evaluator, H, x)
    start = time()
    MOI.eval_hessian_objective(evaluator.backend, H, x)
    evaluator.eval_hessian_objective_timer += time() - start
    return
end

function MOI.eval_hessian_constraint(evaluator::Evaluator, H, x, i)
    start = time()
    MOI.eval_hessian_constraint(evaluator.backend, H, x, i)
    evaluator.eval_hessian_constraint_timer += time() - start
    return
end

function MOI.eval_hessian_lagrangian(evaluator::Evaluator, H, x, σ, μ)
    start = time()
    MOI.eval_hessian_lagrangian(evaluator.backend, H, x, σ, μ)
    evaluator.eval_hessian_lagrangian_timer += time() - start
    return
end

function MOI.eval_constraint_jacobian_product(evaluator::Evaluator, y, x, w)
    start = time()
    MOI.eval_constraint_jacobian_product(evaluator.backend, y, x, w)
    evaluator.eval_constraint_jacobian_timer += time() - start
    return
end

function MOI.eval_constraint_jacobian_transpose_product(
    evaluator::Evaluator,
    y,
    x,
    w,
)
    start = time()
    MOI.eval_constraint_jacobian_transpose_product(evaluator.backend, y, x, w)
    evaluator.eval_constraint_jacobian_timer += time() - start
    return
end

function MOI.eval_hessian_lagrangian_product(
    evaluator::Evaluator,
    H,
    x,
    v,
    σ,
    μ,
)
    start = time()
    MOI.eval_hessian_lagrangian_product(evaluator.backend, H, x, v, σ, μ)
    evaluator.eval_hessian_lagrangian_timer += time() - start
    return
end

function add_expression(model::Model, expr)
    push!(model.expressions, parse_expression(model, expr))
    return Nonlinear.ExpressionIndex(length(model.expressions))
end

function eval_univariate_hessian(
    registry::OperatorRegistry,
    id::Integer,
    x::T,
) where {T}
    if id <= registry.univariate_user_operator_start
        ret = Nonlinear._eval_univariate_2nd_deriv(id, x)
        if ret === nothing
            op = registry.univariate_operators[id]
            error("Hessian is not defined for operator $op")
        end
        return ret::T
    end
    offset = id - registry.univariate_user_operator_start
    operator = registry.registered_univariate_operators[offset]
    return eval_univariate_hessian(operator, x)
end

function add_parameter(model::Model, value::Float64)
    push!(model.parameters, value)
    return MOI.Nonlinear.ParameterIndex(length(model.parameters))
end

function Base.getindex(model::Model, p::Nonlinear.ParameterIndex)
    return model.parameters[p.value]
end

function Base.setindex!(model::Model, value::Real, p::Nonlinear.ParameterIndex)
    return model.parameters[p.value] = convert(Float64, value)::Float64
end

function delete(model::Model, c::Nonlinear.ConstraintIndex)
    delete!(model.constraints, c)
    return
end

function Base.getindex(model::Model, index::Nonlinear.ConstraintIndex)
    return model.constraints[index]
end

function MOI.is_valid(model::Model, index::Nonlinear.ConstraintIndex)
    return haskey(model.constraints, index)
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

function eval_multivariate_gradient(
    registry::OperatorRegistry,
    op::Symbol,
    g::AbstractVector{T},
    x::AbstractVector{T},
) where {T}
    @assert length(g) == length(x)
    if op == :+
        fill!(g, one(T))
    elseif op == :-
        g[1] = one(T)
        g[2] = -one(T)
    elseif op == :*
        # Special case performance optimizations for common cases.
        if length(x) == 1
            g[1] = one(T)
        elseif length(x) == 2
            g[1] = x[2]
            g[2] = x[1]
        else
            total = prod(x)
            if iszero(total)
                for i in eachindex(x)
                    g[i] = prod(x[j] for j in eachindex(x) if i != j)
                end
            else
                for i in eachindex(x)
                    g[i] = total / x[i]
                end
            end
        end
    elseif op == :^
        @assert length(x) == 2
        if x[2] == one(T)
            g[1] = one(T)
        elseif x[2] == T(2)
            g[1] = T(2) * x[1]
        else
            g[1] = x[2] * _nan_pow(x[1], x[2] - one(T))
        end
        if x[1] > zero(T)
            g[2] = _nan_pow(x[1], x[2]) * log(x[1])
        else
            g[2] = T(NaN)
        end
    elseif op == :/
        @assert length(x) == 2
        g[1] = one(T) / x[2]
        g[2] = -x[1] / x[2]^2
    elseif op == :ifelse
        @assert length(x) == 3
        g[1] = zero(T)  # It doesn't matter what this is.
        g[2] = x[1] == one(T)
        g[3] = x[1] == zero(T)
    elseif op == :atan
        @assert length(x) == 2
        base = x[1]^2 + x[2]^2
        g[1] = x[2] / base
        g[2] = -x[1] / base
    elseif op == :min
        fill!(g, zero(T))
        _, i = findmin(x)
        g[i] = one(T)
    elseif op == :max
        fill!(g, zero(T))
        _, i = findmax(x)
        g[i] = one(T)
    else
        id = registry.multivariate_operator_to_id[op]
        offset = id - registry.multivariate_user_operator_start
        operator = registry.registered_multivariate_operators[offset]
        @assert length(x) == operator.N
        operator.∇f(g, x)
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

function eval_comparison_function(
    ::OperatorRegistry,
    op::Symbol,
    lhs::T,
    rhs::T,
)::Bool where {T}
    if op == :<=
        return lhs <= rhs
    elseif op == :>=
        return lhs >= rhs
    elseif op == :(==)
        return lhs == rhs
    elseif op == :<
        return lhs < rhs
    else
        @assert op == :>
        return lhs > rhs
    end
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

function MOI.features_available(d::NLPEvaluator)
    # Check if we are missing any hessians for user-defined operators, in which
    # case we need to disable :Hess and :HessVec.
    d.disable_2ndorder =
        any(_no_hessian, d.data.operators.registered_univariate_operators) ||
        any(_no_hessian, d.data.operators.registered_multivariate_operators)
    if d.disable_2ndorder
        return [:Grad, :Jac, :JacVec]
    end
    return [:Grad, :Jac, :JacVec, :Hess, :HessVec]
end
