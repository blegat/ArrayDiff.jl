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

function Evaluator(
    model::ArrayDiff.Model,
    ::Mode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    return Evaluator(model, NLPEvaluator(model, ordered_variables))
end

# Called by solvers (e.g., NLopt) via:
#   MOI.Nonlinear.Evaluator(nlp_model, ad_backend, vars)
# When nlp_model is an ArrayNonlinearFunction and ad_backend is Mode(),
# we build an ArrayDiff.Model and return our Evaluator.
function Nonlinear.Evaluator(
    func::ArrayNonlinearFunction,
    ::Mode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    ad_model = Model()
    set_objective(ad_model, func)
    return Evaluator(ad_model, NLPEvaluator(ad_model, ordered_variables))
end

include("JuMP/JuMP.jl")

end  # module
