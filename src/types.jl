# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    struct Expression
        nodes::Vector{Node}
        values::Vector{Float64}
    end

The core type that represents a nonlinear expression. See the MathOptInterface
documentation for information on how the nodes and values form an expression
tree.
"""
struct Expression{T}
    nodes::Vector{Node}
    values::Vector{T}
    Expression{T}() where {T} = new{T}(Node[], T[])
end

function Base.:(==)(x::Expression, y::Expression)
    return x.nodes == y.nodes && x.values == y.values
end

"""
    struct Constraint
        expression::Expression
        set::Union{
            MOI.LessThan{Float64},
            MOI.GreaterThan{Float64},
            MOI.EqualTo{Float64},
            MOI.Interval{Float64},
        }
    end

A type to hold information relating to the nonlinear constraint `f(x) in S`,
where `f(x)` is defined by `.expression`, and `S` is `.set`.
"""
struct Constraint{T}
    expression::Expression{T}
    set::Union{
        MOI.LessThan{T},
        MOI.GreaterThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    }
end

"""
    ParameterIndex

An index to a nonlinear parameter that is returned by [`add_parameter`](@ref).
Given `data::Model` and `p::ParameterIndex`, use `data[p]` to retrieve
the current value of the parameter and `data[p] = value` to set a new value.
"""
struct ParameterIndex
    value::Int
end

"""
    ExpressionIndex

An index to a nonlinear expression that is returned by [`add_expression`](@ref).

Given `data::Model` and `ex::ExpressionIndex`, use `data[ex]` to
retrieve the corresponding [`Expression`](@ref).
"""
struct ExpressionIndex
    value::Int
end

"""
    ConstraintIndex

An index to a nonlinear constraint that is returned by [`add_constraint`](@ref).

Given `data::Model` and `c::ConstraintIndex`, use `data[c]` to
retrieve the corresponding [`Constraint`](@ref).
"""
struct ConstraintIndex
    value::Int
end

# We don't need to store the full vector of `linearity` but we return
# it because it is needed in `compute_hessian_sparsity`.
function _subexpression_and_linearity(
    expr::Expression,
    moi_index_to_consecutive_index,
    partials_storage_ϵ::Vector{Float64},
    d,
)
    nodes = _replace_moi_variables(expr.nodes, moi_index_to_consecutive_index)
    adj = adjacency_matrix(nodes)
    linearity = if d.want_hess
        _classify_linearity(nodes, adj, d.subexpression_linearity)
    else
        [NONLINEAR]
    end
    return _SubexpressionStorage(
        nodes,
        adj,
        convert(Vector{Float64}, expr.values),
        partials_storage_ϵ,
        linearity[1],
    ),
    linearity
end

struct _FunctionStorage
    expr::_SubexpressionStorage
    grad_sparsity::Vector{Int}
    # Nonzero pattern of Hessian matrix
    hess_I::Vector{Int}
    hess_J::Vector{Int}
    rinfo::Coloring.RecoveryInfo # coloring info for hessians
    seed_matrix::Matrix{Float64}
    # subexpressions which this function depends on, ordered for forward pass.
    dependent_subexpressions::Vector{Int}

    function _FunctionStorage(
        expr::_SubexpressionStorage,
        num_variables,
        coloring_storage::Coloring.IndexedSet,
        want_hess::Bool,
        subexpressions::Vector{_SubexpressionStorage},
        dependent_subexpressions,
        subexpression_edgelist,
        subexpression_variables,
        linearity::Vector{Linearity},
    )
        empty!(coloring_storage)
        _compute_gradient_sparsity!(coloring_storage, expr.nodes)
        for k in dependent_subexpressions
            _compute_gradient_sparsity!(
                coloring_storage,
                subexpressions[k].nodes,
            )
        end
        grad_sparsity = sort!(collect(coloring_storage))
        empty!(coloring_storage)
        if want_hess
            edgelist = _compute_hessian_sparsity(
                expr.nodes,
                expr.adj,
                linearity,
                subexpression_edgelist,
                subexpression_variables,
            )
            hess_I, hess_J, rinfo = Coloring.hessian_color_preprocess(
                edgelist,
                num_variables,
                coloring_storage,
            )
            seed_matrix = Coloring.seed_matrix(rinfo)
            return new(
                expr,
                grad_sparsity,
                hess_I,
                hess_J,
                rinfo,
                seed_matrix,
                dependent_subexpressions,
            )
        else
            return new(
                expr,
                grad_sparsity,
                Int[],
                Int[],
                Coloring.RecoveryInfo(),
                Array{Float64}(undef, 0, 0),
                dependent_subexpressions,
            )
        end
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
mutable struct Model{T}
    objective::Union{Nothing,Expression{T}}
    expressions::Vector{Expression{T}}
    constraints::OrderedCollections.OrderedDict{ConstraintIndex,Constraint{T}}
    parameters::Vector{T}
    operators::OperatorRegistry
    # This is a private field, used only to increment the ConstraintIndex.
    last_constraint_index::Int64
    function Model{T}() where {T}
        return new{T}(
            nothing,
            Expression{T}[],
            OrderedCollections.OrderedDict{ConstraintIndex,Constraint{T}}(),
            T[],
            OperatorRegistry(),
            0,
        )
    end
end

Model() = Model{Float64}()

mutable struct Evaluator{T,B} <: MOI.AbstractNLPEvaluator
    # The internal datastructure.
    model::Model{T}
    # The abstract-differentiation backend
    backend::B
    # ordered_constraints is needed because `OrderedDict` doesn't support
    # looking up a key by the linear index.
    ordered_constraints::Vector{ConstraintIndex}
    # Storage for the NLPBlockDual, so that we can query the dual of individual
    # constraints without needing to query the full vector each time.
    constraint_dual::Vector{T}
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
        model::Model{T},
        backend::B = nothing,
    ) where {T,B<:Union{Nothing,MOI.AbstractNLPEvaluator}}
        return new{T,B}(
            model,
            backend,
            MOI.ConstraintIndex[],
            T[],
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

_bound(s::MOI.LessThan) = MOI.NLPBoundsPair(-Inf, s.upper)
_bound(s::MOI.GreaterThan) = MOI.NLPBoundsPair(s.lower, Inf)
_bound(s::MOI.EqualTo) = MOI.NLPBoundsPair(s.value, s.value)
_bound(s::MOI.Interval) = MOI.NLPBoundsPair(s.lower, s.upper)

"""
    NLPEvaluator(
        model::Nonlinear.Model,
        ordered_variables::Vector{MOI.VariableIndex},
    )

Return an `NLPEvaluator` object that implements the `MOI.AbstractNLPEvaluator`
interface.

!!! warning
    Before using, you must initialize the evaluator using `MOI.initialize`.
"""
mutable struct NLPEvaluator <: MOI.AbstractNLPEvaluator
    data::Model
    ordered_variables::Vector{MOI.VariableIndex}

    objective::Union{Nothing,_FunctionStorage}
    constraints::Vector{_FunctionStorage}
    subexpressions::Vector{_SubexpressionStorage}
    subexpression_order::Vector{Int}
    # Storage for the subexpressions in reverse-mode automatic differentiation.
    subexpression_forward_values::Vector{Float64}
    subexpression_reverse_values::Vector{Float64}
    subexpression_linearity::Vector{Linearity}

    # A cache of the last x. This is used to guide whether we need to re-run
    # reverse-mode automatic differentiation.
    last_x::Vector{Float64}

    # Temporary storage for computing Jacobians. This is also used as temporary
    # storage for the input of multivariate functions.
    jac_storage::Vector{Float64}
    # Temporary storage for the gradient of multivariate functions
    user_output_buffer::Vector{Float64}

    # storage for computing hessians
    # these Float64 vectors are reinterpreted to hold multiple epsilon components
    # so the length should be multiplied by the maximum number of epsilon components
    disable_2ndorder::Bool # don't offer Hess or HessVec
    want_hess::Bool
    storage_ϵ::Vector{Float64} # (longest expression including subexpressions)
    input_ϵ::Vector{Float64} # (number of variables)
    output_ϵ::Vector{Float64} # (number of variables)
    subexpression_forward_values_ϵ::Vector{Float64} # (number of subexpressions)
    subexpression_reverse_values_ϵ::Vector{Float64} # (number of subexpressions)
    hessian_sparsity::Vector{Tuple{Int64,Int64}}
    max_chunk::Int # chunk size for which we've allocated storage

    function NLPEvaluator(
        data::Model,
        ordered_variables::Vector{MOI.VariableIndex},
    )
        return new(data, ordered_variables)
    end
end
