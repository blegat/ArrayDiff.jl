module TestMathOptAIExt

using JuMP
using Test

import ArrayDiff
import Ipopt
import MathOptAI
import MathOptInterface as MOI

is_test(x) = startswith(string(x), "test_")

function runtests()
    @testset "$name" for name in filter(is_test, names(@__MODULE__; all = true))
        getfield(@__MODULE__, name)()
    end
    return
end

# Build a tiny `ArrayDiff.Evaluator` whose residual is `[x1 + x1*x2, x2 - x1*x2]`.
# This is the analogue of a trained NN: a multi-output Julia function whose
# Jacobian we want ArrayDiff to compute inside the `MOI.VectorNonlinearOracle`.
function _build_test_evaluator()
    ad_model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    e = ArrayDiff.add_expression(ad_model, :($x1 * $x2))
    ArrayDiff.set_residual!(ad_model, :([$x1 + $e, $x2 - $e]))
    evaluator = ArrayDiff.Evaluator(ad_model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad, :Jac, :JacVec])
    return evaluator
end

function test_build_predictor_only_gray_box()
    evaluator = _build_test_evaluator()
    # Default is `gray_box = true`; the non-gray-box path is unsupported.
    @test_throws ErrorException MathOptAI.build_predictor(
        evaluator;
        gray_box = false,
    )
    @test_throws ErrorException MathOptAI.build_predictor(
        evaluator;
        gray_box = true,
        hessian = true,
    )
    predictor = MathOptAI.build_predictor(evaluator)
    @test predictor isa MathOptAI.GrayBox{<:ArrayDiff.Evaluator}
    @test predictor.hessian == false
    return
end

function test_vector_nonlinear_oracle()
    evaluator = _build_test_evaluator()
    predictor = MathOptAI.build_predictor(evaluator; gray_box = true)
    oracle = MOI.VectorNonlinearOracle(predictor, 2)
    # f(x) - y = 0, with f₁ = x₁ + x₁x₂ and f₂ = x₂ - x₁x₂.
    # Hand-evaluate at x = (3, 4), y = (15, -8): residual must be zero.
    ret = zeros(2)
    oracle.eval_f(ret, [3.0, 4.0, 15.0, -8.0])
    @test ret == [0.0, 0.0]
    # Jacobian at (3, 4): ∂f/∂x = [1+x₂ x₁; -x₂ 1-x₁] = [5 3; -4 -2].
    # Flat storage is column-major over (r, c), then the -I block for y.
    J = zeros(2 * 2 + 2)
    oracle.eval_jacobian(J, [3.0, 4.0, 15.0, -8.0])
    @test J ≈ [5.0, -4.0, 3.0, -2.0, -1.0, -1.0]
    return
end

function test_end_to_end_with_ipopt()
    evaluator = _build_test_evaluator()
    # Min ||y - [10, 0]||² subject to y = f(x), x free.
    # f₁ = x₁(1 + x₂), f₂ = x₂(1 - x₁). One feasible solution to y = (10, 0)
    # is x = (10, 0) → f = (10, 0). Optimization should drive y → (10, 0).
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x[1:2])
    set_start_value(x[1], 1.0)
    set_start_value(x[2], 0.5)
    y, _ = MathOptAI.add_predictor(model, evaluator, x; gray_box = true)
    @objective(model, Min, (y[1] - 10.0)^2 + y[2]^2)
    optimize!(model)
    assert_is_solved_and_feasible(model)
    @test isapprox(value(y[1]), 10.0; atol = 1e-4)
    @test isapprox(value(y[2]), 0.0; atol = 1e-4)
    return
end

end  # module

TestMathOptAIExt.runtests()
