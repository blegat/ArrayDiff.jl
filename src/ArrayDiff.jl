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
import SparseMatrixColorings

"""
    Mode(coloring_algorithm::SparseMatrixColorings.GreedyColoringAlgorithm) <: AbstractAutomaticDifferentiation

Fork of `MOI.Nonlinear.SparseReverseMode` to add array support.
"""
struct Mode{C<:SparseMatrixColorings.GreedyColoringAlgorithm} <:
       MOI.Nonlinear.AbstractAutomaticDifferentiation
    coloring_algorithm::C
end

function Mode()
    return Mode(
        SparseMatrixColorings.GreedyColoringAlgorithm(;
            decompression = :substitution,
        ),
    )
end

function MOI.Nonlinear.Evaluator(
    model::MOI.Nonlinear.Model,
    mode::Mode,
    ordered_variables::Vector{MOI.VariableIndex},
)
    return MOI.Nonlinear.Evaluator(
        model,
        NLPEvaluator(model, ordered_variables, mode.coloring_algorithm),
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

include("coloring.jl")
include("graph_tools.jl")
include("sizes.jl")
include("types.jl")
include("utils.jl")

include("reverse_mode.jl")
include("forward_over_reverse.jl")
include("mathoptinterface_api.jl")

end  # module
