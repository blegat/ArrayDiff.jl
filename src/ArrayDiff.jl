# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ArrayDiff

import ChainRulesCore
import ForwardDiff
import LinearAlgebra
import MathOptInterface as MOI
const Nonlinear = MOI.Nonlinear
import SparseArrays
import OrderedCollections

"""
    Mode{S}() <: MOI.Nonlinear.AbstractAutomaticDifferentiation

Fork of `MOI.Nonlinear.SparseReverseMode` to add array support.

The type parameter `S` is the storage type used for the AD tape (forward,
partials, and reverse storage of each subexpression). It must satisfy
`S<:AbstractVector{<:Real}`. Defaults to `Vector{Float64}`. Pass a different
`S` (for example `Vector{Float32}` or `CuVector{Float64}`) to run AD in
another precision or keep the tape on a GPU.
"""
struct Mode{S<:AbstractVector{<:Real}} <:
       MOI.Nonlinear.AbstractAutomaticDifferentiation end

Mode() = Mode{Vector{Float64}}()

# Override basic math functions to return NaN instead of throwing errors.
# This is what NLP solvers expect, and sometimes the results aren't needed
# anyway, because the code may compute derivatives wrt constants.
import NaNMath:
    sin,
    cos,
    tan,
    asin,
    acos,
    acosh,
    atanh,
    log,
    log2,
    log10,
    lgamma,
    log1p,
    pow,
    sqrt

include("Coloring/Coloring.jl")
include("graph_tools.jl")
include("sizes.jl")
include("univariate_expressions.jl")
include("operators.jl")
include("types.jl")
include("utils.jl")

include("reverse_mode.jl")
include("forward_over_reverse.jl")
include("mathoptinterface_api.jl")
include("model.jl")
include("parse.jl")
include("evaluator.jl")

include("array_nonlinear_function.jl")
include("parse_moi.jl")

"""
    from_onnx(model; inputs)

Translate an ONNX `ModelProto` into a Julia `Expr` (or `Dict{String,Expr}` for
multi-output graphs) suitable for `ArrayDiff.set_objective` or for composing
further with `sum`, `LinearAlgebra.norm`, etc.

`inputs` maps each ONNX graph-input name to the Julia value that should stand
in for it — typically a `Vector{MOI.VariableIndex}` or a
`Matrix{MOI.VariableIndex}` of the appropriate shape. Initializer tensors are
inlined as `Vector{Float64}` / `Matrix{Float64}` constants.

The implementation lives in the package extension `ArrayDiffONNXExt`, which is
loaded automatically once `ONNX` is imported alongside `ArrayDiff`.
"""
function from_onnx end

model(::Mode{S}) where {S} = Model{eltype(S)}()

# Hook so that solvers using `MOI.Nonlinear.model(backend)` (for example,
# NLopt and NLPModelsJuMP) receive an ArrayDiff model for an ArrayDiff mode.
# ArrayDiff handles scalar nonlinear functions. The outer layers own quadratic
# functions, variables, bounds, parameters, and vector nonlinear oracles.
function Nonlinear.model(mode::Mode)
    return Nonlinear.ModelWithQuad(
        Nonlinear.ModelWithOracles(model(mode)),
    )
end

# Extend MOI.Nonlinear.set_objective so that solvers calling
# MOI.Nonlinear.set_objective(arraydiff_model, snf) dispatch here.
function Nonlinear.set_objective(model::Model, obj::MOI.ScalarNonlinearFunction)
    model.objective = parse_expression(model, obj)
    if model.objective_sense == MOI.FEASIBILITY_SENSE
        model.objective_sense = MOI.MIN_SENSE
    end
    return
end

function Nonlinear.set_objective(model::Model, ::Nothing)
    model.objective = nothing
    return
end

Nonlinear._parameter_values(model::Model) = model.parameters
Nonlinear._has_nonlinear_data(model::Model) =
    model.objective !== nothing || !isempty(model.constraints)
Nonlinear._is_nonlinear_input(
    ::Model{T},
    ::MOI.ScalarNonlinearFunction,
    ::Union{
        MOI.LessThan{T},
        MOI.GreaterThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    },
) where {T} = true
Nonlinear._is_nonlinear_objective(
    ::Model,
    ::MOI.ScalarNonlinearFunction,
) = true

MOI.supports_incremental_interface(::Model) = true
MOI.supports(::Model, ::MOI.ObjectiveSense) = true
MOI.get(model::Model, ::MOI.ObjectiveSense) = model.objective_sense
function MOI.set(model::Model, ::MOI.ObjectiveSense, sense)
    model.objective_sense = sense
    return
end
MOI.supports(
    ::Model,
    ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
) = true
function MOI.set(
    model::Model,
    ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    f::MOI.ScalarNonlinearFunction,
)
    return Nonlinear.set_objective(model, f)
end

function MOI.supports_constraint(
    ::Model{T},
    ::Type{MOI.ScalarNonlinearFunction},
    ::Type{S},
) where {
    T,
    S<:Union{
        MOI.LessThan{T},
        MOI.GreaterThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    },
}
    return true
end
function MOI.add_constraint(
    model::Model{T},
    f::MOI.ScalarNonlinearFunction,
    s::S,
) where {
    T,
    S<:Union{
        MOI.LessThan{T},
        MOI.GreaterThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    },
}
    ci = add_constraint(model, f, s)
    return MOI.ConstraintIndex{typeof(f),S}(ci.value)
end
function MOI.is_valid(
    model::Model{T},
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,S},
) where {
    T,
    S<:Union{
        MOI.LessThan{T},
        MOI.GreaterThan{T},
        MOI.EqualTo{T},
        MOI.Interval{T},
    },
}
    return haskey(model.constraints, ConstraintIndex(ci.value))
end

function Nonlinear.constraint_rows(
    model::Model,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:MOI.AbstractScalarSet},
)
    row = findfirst(==(ConstraintIndex(ci.value)), keys(model.constraints))
    return [something(row)]
end
Nonlinear.constraint_dual_starts(model::Model{T}) where {T} =
    fill(nothing, length(model.constraints))

# Create an ArrayDiff Evaluator from an ArrayDiff Model.
function Evaluator(
    model::ArrayDiff.Model,
    ::Mode{S},
    ordered_variables::Vector{MOI.VariableIndex},
) where {S<:AbstractVector{<:Real}}
    return Evaluator(model, NLPEvaluator{eltype(S),S}(model, ordered_variables))
end

# Called by solvers via MOI.Nonlinear.Evaluator(nlp_model, ad_backend, vars).
# When nlp_model is an ArrayDiff.Model (created by model(::Mode)),
# the model already has the parsed objective — just build the evaluator.
function Nonlinear.Evaluator(
    model::ArrayDiff.Model,
    mode::Mode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    return Evaluator(model, mode, ordered_variables)
end

function Nonlinear._constraint_bounds(evaluator::Evaluator)
    return [_bound(c.set) for (_, c) in evaluator.model.constraints]
end
Nonlinear._has_objective(evaluator::Evaluator) =
    evaluator.model.objective !== nothing

include("JuMP/JuMP.jl")

end  # module
