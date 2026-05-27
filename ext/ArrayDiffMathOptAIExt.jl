# Copyright (c) 2026: Sophie Lequeu, Benoît Legat
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module ArrayDiffMathOptAIExt

import ArrayDiff
import MathOptAI
import MathOptInterface as MOI

"""
    MathOptAI.build_predictor(
        predictor::ArrayDiff.Evaluator;
        gray_box::Bool = true,
        hessian::Bool = false,
    )

Wrap an `ArrayDiff.Evaluator` whose residual has been set via
`ArrayDiff.set_residual!` as a [`MathOptAI.GrayBox`](@ref) predictor. The
evaluator must already have been initialized with
`MOI.initialize(evaluator, [:Jac, :JacVec])` (or a superset). ArrayDiff is then
used to compute the residual and its Jacobian-vector products in the resulting
[`MOI.VectorNonlinearOracle`](@ref) — the analogue of the PyTorch extension's
`torch.func.jacrev`, but driven by ArrayDiff's reverse-mode tape.

Only `gray_box = true` is supported; `hessian = true` is not yet implemented.
"""
function MathOptAI.build_predictor(
    predictor::ArrayDiff.Evaluator;
    gray_box::Bool = true,
    hessian::Bool = false,
)
    @assert gray_box "only `gray_box = true` is supported for `ArrayDiff.Evaluator`"
    @assert !hessian "`hessian = true` is not yet supported for `ArrayDiff.Evaluator`"
    return MathOptAI.GrayBox(predictor; hessian = false)
end

function MOI.VectorNonlinearOracle(
    predictor::MathOptAI.GrayBox{<:ArrayDiff.Evaluator},
    input_dimension::Int,
)
    evaluator = predictor.predictor
    output_dimension = ArrayDiff.residual_dimension(evaluator)
    # We model the function as:
    #     0 <= F(x) - y <= 0
    function eval_f(ret::AbstractVector, x::AbstractVector)
        ArrayDiff.eval_residual!(
            evaluator,
            view(ret, 1:output_dimension),
            view(x, 1:input_dimension),
        )
        for i in 1:output_dimension
            ret[i] -= x[input_dimension+i]
        end
        return
    end
    # Note the order of the for-loops, first over the output_dimension, and then
    # across the input_dimension. This makes the Jacobian structure of ∇F(x) be
    # column-major and dense with respect to x.
    jacobian_structure = Tuple{Int,Int}[
        (r, c) for c in 1:input_dimension for r in 1:output_dimension
    ]
    # We also need to add non-zero terms for the `-I` component of the Jacobian.
    for i in 1:output_dimension
        push!(jacobian_structure, (i, input_dimension + i))
    end
    function eval_jacobian(ret::AbstractVector, x::AbstractVector)
        input = view(x, 1:input_dimension)
        seed = zeros(output_dimension)
        row = zeros(input_dimension)
        # Reverse-mode: one J'v call per output row gives that row of the
        # Jacobian. Matches ArrayDiff's reverse-mode tape orientation.
        for r in 1:output_dimension
            fill!(seed, 0.0)
            seed[r] = 1.0
            ArrayDiff.eval_residual_jtprod!(evaluator, row, input, seed)
            for c in 1:input_dimension
                ret[(c-1)*output_dimension+r] = row[c]
            end
        end
        for i in 1:output_dimension
            ret[input_dimension*output_dimension+i] = -1.0
        end
        return
    end
    return MOI.VectorNonlinearOracle(;
        dimension = input_dimension + output_dimension,
        l = zeros(output_dimension),
        u = zeros(output_dimension),
        eval_f,
        jacobian_structure,
        eval_jacobian,
        hessian_lagrangian_structure = Tuple{Int,Int}[],
        eval_hessian_lagrangian = nothing,
    )
end

end  # module ArrayDiffMathOptAIExt
