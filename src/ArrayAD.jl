# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ArrayAD

import ForwardDiff
import MathOptInterface as MOI
const Nonlinear = MOI.Nonlinear
import SparseArrays

"""
    Mode() <: AbstractAutomaticDifferentiation

Fork of `MOI.Nonlinear.SparseReverseMode` to add array support.
"""
struct Mode <: MOI.Nonlinear.AbstractAutomaticDifferentiation end

function MOI.Nonlinear.Evaluator(
    model::MOI.Nonlinear.Model,
    ::Mode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    return MOI.Nonlinear.Evaluator(
        model,
        NLPEvaluator(model, ordered_variables),
    )
end

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
include("types.jl")
include("utils.jl")

include("reverse_mode.jl")
include("forward_over_reverse.jl")
include("mathoptinterface_api.jl")

end  # module
