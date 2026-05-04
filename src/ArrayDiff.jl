# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ArrayDiff

import ForwardDiff
import MathOptInterface as MOI
const Nonlinear = MOI.Nonlinear
import SparseArrays
import OrderedCollections

"""
    Mode() <: MOI.Nonlinear.AbstractAutomaticDifferentiation

Fork of `MOI.Nonlinear.SparseReverseMode` to add array support.
"""
struct Mode <: MOI.Nonlinear.AbstractAutomaticDifferentiation end

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

model(::Mode) = Model()

# Extend MOI.Nonlinear.set_objective so that solvers calling
# MOI.Nonlinear.set_objective(arraydiff_model, snf) dispatch here.
function Nonlinear.set_objective(model::Model, obj::MOI.ScalarNonlinearFunction)
    model.objective = parse_expression(model, obj)
    return
end

function Nonlinear.set_objective(model::Model, ::Nothing)
    model.objective = nothing
    return
end

# Create an ArrayDiff Evaluator from an ArrayDiff Model.
function Evaluator(
    model::ArrayDiff.Model,
    ::Mode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    return Evaluator(model, NLPEvaluator(model, ordered_variables))
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

include("JuMP/JuMP.jl")

end  # module
