module TestArrayDiff

using Test
import LinearAlgebra
import SparseArrays

import MathOptInterface as MOI
const Nonlinear = MOI.Nonlinear

import ArrayDiff
const Coloring = ArrayDiff.Coloring

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_objective_quadratic_univariate()
    x = MOI.VariableIndex(1)
    scalar = Nonlinear.Model()
    Nonlinear.set_objective(model, :($x * $x))
    vector = Nonlinear.Model()
    Nonlinear.set_objective(vector, :([$x] * [$x]))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x])
    MOI.initialize(evaluator, [:Grad, :Jac, :Hess])
    @test MOI.eval_objective(evaluator, [1.2]) == 1.2^2 + 1
    g = [NaN]
    MOI.eval_objective_gradient(evaluator, g, [1.2])
    @test g == [2.4]
    @test MOI.hessian_objective_structure(evaluator) == [(1, 1)]
    H = [NaN]
    MOI.eval_hessian_objective(evaluator, H, [1.2])
    @test H == [2.0]
    @test MOI.hessian_lagrangian_structure(evaluator) == [(1, 1)]
    H = [NaN]
    MOI.eval_hessian_lagrangian(evaluator, H, [1.2], 1.5, Float64[])
    @test H == 1.5 .* [2.0]
    MOI.eval_hessian_lagrangian_product(
        evaluator,
        H,
        [1.2],
        [1.2],
        1.5,
        Float64[],
    )
    @test H == [1.5 * 2.0 * 1.2]
    return
end

end  # module

TestArrayDiff.runtests()

import MathOptInterface as MOI
import ArrayDiff
using Test
const Nonlinear = MOI.Nonlinear
model = Nonlinear.Model()
x = MOI.VariableIndex(1)
Nonlinear.set_objective(model, :(dot([$x], [$x])))
evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x])
MOI.initialize(evaluator, [:Grad])
sizes = evaluator.backend.objective.expr.sizes
@test MOI.eval_objective(evaluator, [1.2]) == 1.2^2
@test sizes.ndims == [0, 1, 0, 1, 0]
@test sizes.size_offset == [0, 1, 0, 0, 0]
@test sizes.size == [1, 1]
@test sizes.storage_offset == [0, 1, 2, 3, 4, 5]

y = MOI.VariableIndex(1)
Nonlinear.set_objective(model, :(dot([$x, $y] - [1, 2], -[1, 2] + [$x, $y])))
MOI.initialize(evaluator, [:Grad])
sizes = evaluator.backend.objective.expr.sizes
@test MOI.eval_objective(evaluator, [1.2]) == 1.2^2
@test sizes.ndims == [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
@test sizes.size_offset == [0, 6, 5, 0, 0, 4, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0]
@test sizes.size == [2, 2, 2, 2, 2, 2, 2]
@test sizes.storage_offset ==
      [0, 1, 3, 5, 6, 7, 9, 10, 11, 13, 15, 17, 18, 19, 21, 22, 23]
g = [NaN]
MOI.eval_objective_gradient(evaluator, g, [1.2])
