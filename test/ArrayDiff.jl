module TestArrayDiff

using Test
import LinearAlgebra
import SparseArrays

import MathOptInterface as MOI
const Nonlinear = MOI.Nonlinear

import ArrayDiff
const Coloring = ArrayDiff.Coloring

# Wrapped in a typed function so `@allocated` doesn't capture the
# return-value boxing that happens when calling `eval_objective`
# directly in tests where the local variable holding the input has been
# reassigned to a different type (and is therefore `Any`-typed at the
# call site).
_obj(ev, x) = MOI.eval_objective(ev, x)

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

function test_objective_dot_univariate()
    model = ArrayDiff.Model()
    x = MOI.VariableIndex(1)
    ArrayDiff.set_objective(model, :(dot([$x], [$x])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x])
    MOI.initialize(evaluator, [:Grad, :Hess])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 1, 0, 1, 0]
    @test sizes.size_offset == [0, 1, 0, 0, 0]
    @test sizes.size == [1, 1]
    @test sizes.storage_offset == [0, 1, 2, 3, 4, 5]
    x = [1.2]
    @test MOI.eval_objective(evaluator, x) == x[1]^2
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(1)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g[1] == 2x[1]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_dot_univariate_and_scalar_mult()
    model = ArrayDiff.Model()
    x = MOI.VariableIndex(1)
    ArrayDiff.set_objective(model, :(2*(dot([$x], [$x]))))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 0, 0, 1, 0, 1, 0]
    @test sizes.size_offset == [0, 0, 0, 1, 0, 0, 0]
    @test sizes.size == [1, 1]
    @test sizes.storage_offset == [0, 1, 2, 3, 4, 5, 6, 7]
    x = [1.2]
    @test MOI.eval_objective(evaluator, x) == 2*x[1]^2
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(1)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g[1] == 4x[1]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_dot_bivariate()
    model = ArrayDiff.Model()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    ArrayDiff.set_objective(
        model,
        :(dot([$x, $y] - [1, 2], -[1, 2] + [$x, $y])),
    )
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x, y])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
    @test sizes.size_offset == [0, 6, 5, 0, 0, 4, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0]
    @test sizes.size == [2, 2, 2, 2, 2, 2, 2]
    @test sizes.storage_offset ==
          [0, 1, 3, 5, 6, 7, 9, 10, 11, 13, 15, 17, 18, 19, 21, 22, 23]
    x = [5.0, -1.0]
    @test MOI.eval_objective(evaluator, x) ≈ 25
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == 2(x - [1, 2])
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_hcat_scalars()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(model, :(dot([$x1 $x3], [$x2 $x4])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 0, 0, 2, 0, 0]
    @test sizes.size_offset == [0, 2, 0, 0, 0, 0, 0]
    @test sizes.size == [1, 2, 1, 2]
    @test sizes.storage_offset == [0, 1, 3, 4, 5, 7, 8, 9]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == 14.0
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [2.0, 1.0, 4.0, 3.0]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_hcat_vectors()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(
        model,
        :(dot(hcat([$x1], [$x3]), hcat([$x2], [$x4]))),
    )
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 1, 0, 1, 0, 2, 1, 0, 1, 0]
    @test sizes.size_offset == [0, 6, 5, 0, 4, 0, 2, 1, 0, 0, 0]
    @test sizes.size == [1, 1, 1, 2, 1, 1, 1, 2]
    @test sizes.storage_offset == [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == 14.0
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [2.0, 1.0, 4.0, 3.0]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_dot_bivariate_on_rows()
    model = ArrayDiff.Model()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    ArrayDiff.set_objective(model, :(dot([$x $y] - [1 2], -[1 2] + [$x $y])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x, y])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 2, 0, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0]
    @test sizes.size_offset ==
          [0, 12, 10, 0, 0, 8, 0, 0, 6, 4, 2, 0, 0, 0, 0, 0]
    @test sizes.size == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    @test sizes.storage_offset ==
          [0, 1, 3, 5, 6, 7, 9, 10, 11, 13, 15, 17, 18, 19, 21, 22, 23]
    x = [5.0, -1.0]
    @test MOI.eval_objective(evaluator, x) ≈ 25
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == 2(x - [1, 2])
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_univariate()
    model = ArrayDiff.Model()
    x = MOI.VariableIndex(1)
    ArrayDiff.set_objective(model, :(norm([$x])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 1, 0]
    @test sizes.size_offset == [0, 0, 0]
    @test sizes.size == [1]
    @test sizes.storage_offset == [0, 1, 2, 3]
    x = [1.2]
    @test MOI.eval_objective(evaluator, x) == abs(x[1])
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(1)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g[1] == sign(x[1])
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_bivariate()
    model = ArrayDiff.Model()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    ArrayDiff.set_objective(model, :(norm([$x, $y])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x, y])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 1, 0, 0]
    @test sizes.size_offset == [0, 0, 0, 0]
    @test sizes.size == [2]
    @test sizes.storage_offset == [0, 1, 3, 4, 5]
    x = [3.0, 4.0]
    @test MOI.eval_objective(evaluator, x) == 5.0
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == x / 5.0
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    y = [0.0, 0.0]
    @test MOI.eval_objective(evaluator, y) == 0.0
    @test 0 == @allocated _obj(evaluator, y)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, y)
    @test g == [0.0, 0.0]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, y)
    return
end

function test_objective_norm_of_row_vector()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    ArrayDiff.set_objective(model, :(norm([$x1 $x2])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 0, 0]
    @test sizes.size_offset == [0, 0, 0, 0]
    @test sizes.size == [1, 2]
    @test sizes.storage_offset == [0, 1, 3, 4, 5]
    x1 = 1.0
    x2 = 2.0
    x = [x1, x2]
    @test MOI.eval_objective(evaluator, x) == sqrt(5.0)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [1.0 / sqrt(5.0), 2.0 / sqrt(5.0)]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_of_vcat_vector()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(model, :(norm(vcat($x1, $x3))))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 0, 0]
    @test sizes.size_offset == [0, 0, 0, 0]
    @test sizes.size == [2, 1]
    @test sizes.storage_offset == [0, 1, 3, 4, 5]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == sqrt(10.0)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [1.0 / sqrt(10.0), 0.0, 3.0 / sqrt(10.0), 0.0]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_of_vcat_matrix()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(model, :(norm(vcat([$x1 $x3], [$x2 $x4]))))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 2, 0, 0, 2, 0, 0]
    @test sizes.size_offset == [0, 4, 2, 0, 0, 0, 0, 0]
    @test sizes.size == [1, 2, 1, 2, 2, 2]
    @test sizes.storage_offset == [0, 1, 5, 7, 8, 9, 11, 12, 13]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == sqrt(30.0)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [
        1.0 / sqrt(30.0),
        2.0 / sqrt(30.0),
        3.0 / sqrt(30.0),
        4.0 / sqrt(30.0),
    ]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_of_row()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    ArrayDiff.set_objective(model, :(norm(row($x1, $x2))))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 0, 0]
    @test sizes.size_offset == [0, 0, 0, 0]
    @test sizes.size == [1, 2]
    @test sizes.storage_offset == [0, 1, 3, 4, 5]
    x1 = 1.0
    x2 = 2.0
    x = [x1, x2]
    @test MOI.eval_objective(evaluator, x) == sqrt(5.0)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [1.0 / sqrt(5.0), 2.0 / sqrt(5.0)]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_of_matrix()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(model, :(norm([$x1 $x2; $x3 $x4])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 2, 0, 0, 2, 0, 0]
    @test sizes.size_offset == [0, 4, 2, 0, 0, 0, 0, 0]
    @test sizes.size == [1, 2, 1, 2, 2, 2]
    @test sizes.storage_offset == [0, 1, 5, 7, 8, 9, 11, 12, 13]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == sqrt(30.0)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [
        1.0 / sqrt(30.0),
        2.0 / sqrt(30.0),
        3.0 / sqrt(30.0),
        4.0 / sqrt(30.0),
    ]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_of_matrix_with_sum()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(model, :(norm([$x1 $x2; $x3 $x4] - [1 1; 1 1])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0]
    @test sizes.size_offset ==
          [0, 12, 10, 8, 0, 0, 6, 0, 0, 4, 2, 0, 0, 0, 0, 0]
    @test sizes.size == [1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2]
    @test sizes.storage_offset ==
          [0, 1, 5, 9, 11, 12, 13, 15, 16, 17, 21, 23, 24, 25, 27, 28, 29]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == sqrt(14.0)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [0.0, 1.0 / sqrt(14.0), 2.0 / sqrt(14.0), 3.0 / sqrt(14.0)]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_of_product_of_matrices()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(model, :(norm([$x1 $x2; $x3 $x4] * [1 0; 0 1])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0]
    @test sizes.size_offset ==
          [0, 12, 10, 8, 0, 0, 6, 0, 0, 4, 2, 0, 0, 0, 0, 0]
    @test sizes.size == [1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2]
    @test sizes.storage_offset ==
          [0, 1, 5, 9, 11, 12, 13, 15, 16, 17, 21, 23, 24, 25, 27, 28, 29]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == sqrt(30.0)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [
        1.0 / sqrt(30.0),
        2.0 / sqrt(30.0),
        3.0 / sqrt(30.0),
        4.0 / sqrt(30.0),
    ]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_of_product_of_matrices_with_sum()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(
        model,
        :(norm(([$x1 $x2; $x3 $x4] + [1 1; 1 1]) * [1 0; 0 1])),
    )
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [
        0,
        2,
        2,
        2,
        2,
        0,
        0,
        2,
        0,
        0,
        2,
        2,
        0,
        0,
        2,
        0,
        0,
        2,
        2,
        0,
        0,
        2,
        0,
        0,
    ]
    @test sizes.size_offset == [
        0,
        20,
        18,
        16,
        14,
        0,
        0,
        12,
        0,
        0,
        10,
        8,
        0,
        0,
        6,
        0,
        0,
        4,
        2,
        0,
        0,
        0,
        0,
        0,
    ]
    @test sizes.size ==
          [1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2]
    @test sizes.storage_offset == [
        0,
        1,
        5,
        9,
        13,
        15,
        16,
        17,
        19,
        20,
        21,
        25,
        27,
        28,
        29,
        31,
        32,
        33,
        37,
        39,
        40,
        41,
        43,
        44,
        45,
    ]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == sqrt(54.0)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [
        2.0 / sqrt(54.0),
        3.0 / sqrt(54.0),
        4.0 / sqrt(54.0),
        5.0 / sqrt(54.0),
    ]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_norm_of_mtx_vector_product()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(model, :(norm([$x1 $x2; $x3 $x4] * [1; 1])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0]
    @test sizes.size_offset == [0, 8, 6, 4, 0, 0, 2, 0, 0, 0, 0, 0]
    @test sizes.size == [2, 1, 1, 2, 1, 2, 2, 2, 2, 1]
    @test sizes.storage_offset ==
          [0, 1, 3, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == sqrt(58.0)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [
        3.0 / sqrt(58.0),
        3.0 / sqrt(58.0),
        7.0 / sqrt(58.0),
        7.0 / sqrt(58.0),
    ]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_univariate_operator()
    model = ArrayDiff.Model()
    x = MOI.VariableIndex(1)
    ArrayDiff.set_objective(model, :(sin($x)))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 0]
    @test sizes.size_offset == [0, 0]
    @test sizes.size == []
    @test sizes.storage_offset == [0, 1, 2]
    x = [pi / 4]
    @test MOI.eval_objective(evaluator, x) ≈ sqrt(2) / 2
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(1)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g[1] ≈ cos(pi / 4)
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_broadcasted_product()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(model, :(norm([$x1, $x2] .* [$x3, $x4])))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 1, 0, 0, 1, 0, 0]
    @test sizes.size_offset == [0, 2, 1, 0, 0, 0, 0, 0]
    @test sizes.size == [2, 2, 2, 1]
    @test sizes.storage_offset == [0, 1, 3, 5, 6, 7, 9, 10, 11]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) == sqrt(3.0^2 + 8.0^2)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [9.0, 32.0, 3.0, 16.0] / sqrt(3.0^2 + 8.0^2)
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_broadcasted_matrix_product()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(
        model,
        :(norm([$x1 $x2; $x3 $x4] .* [$x1 $x2; $x3 $x4])),
    )
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0]
    @test sizes.size_offset ==
          [0, 12, 10, 8, 0, 0, 6, 0, 0, 4, 2, 0, 0, 0, 0, 0]
    @test sizes.size == [1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2]
    @test sizes.storage_offset ==
          [0, 1, 5, 9, 11, 12, 13, 15, 16, 17, 21, 23, 24, 25, 27, 28, 29]
    x1 = 1.0
    x2 = 2.0
    x3 = 3.0
    x4 = 4.0
    x = [x1, x2, x3, x4]
    @test MOI.eval_objective(evaluator, x) ==
          sqrt(1.0^2 + 4.0^2 + 9.0^2 + 16.0^2)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [
        2 * 1.0^3 / sqrt(354),
        2 * 2.0^3 / sqrt(354),
        2 * 3.0^3 / sqrt(354),
        2 * 4.0^3 / sqrt(354),
    ]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_broadcasted_tanh()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    ArrayDiff.set_objective(model, :(norm(tanh.([$x1, $x2]))))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 1, 1, 0, 0]
    @test sizes.size_offset == [0, 1, 0, 0, 0]
    @test sizes.size == [2, 2]
    @test sizes.storage_offset == [0, 1, 3, 5, 6, 7]
    x1 = 1.0
    x2 = 2.0
    x = [x1, x2]
    @test MOI.eval_objective(evaluator, x) == sqrt(tanh(1.0)^2 + tanh(2.0)^2)
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g ≈ [
        tanh(1.0) * (1 - tanh(1.0)^2) / sqrt(tanh(1.0)^2 + tanh(2.0)^2),
        tanh(2.0) * (1 - tanh(2.0)^2) / sqrt(tanh(1.0)^2 + tanh(2.0)^2),
    ]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_broadcasted_pow_vector_1()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    ArrayDiff.set_objective(model, :(sum([$x1, $x2] .^ 1)))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad])
    x1v = 3.0
    x2v = -4.0
    x = [x1v, x2v]
    @test MOI.eval_objective(evaluator, x) == x1v + x2v
    @test 0 == @allocated _obj(evaluator, x)
    g = zeros(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == ones(2)
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_broadcasted_pow_vector_2()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    ArrayDiff.set_objective(model, :(sum([$x1, $x2] .^ 2)))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad])
    x1v = 3.0
    x2v = -4.0
    x = [x1v, x2v]
    @test MOI.eval_objective(evaluator, x) == x1v^2 + x2v^2
    @test 0 == @allocated _obj(evaluator, x)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == [2 * x1v, 2 * x2v]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x)
    return
end

function test_objective_broadcasted_pow_matrix_with_constant()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    ArrayDiff.set_objective(
        model,
        :(sum(([$x1 $x2; $x3 $x4] - [1 1; 1 1]) .^ 2)),
    )
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
    MOI.initialize(evaluator, [:Grad])
    xs = [1.0, 2.0, 3.0, 4.0]
    @test MOI.eval_objective(evaluator, xs) ==
          (1-1)^2 + (2-1)^2 + (3-1)^2 + (4-1)^2
    @test 0 == @allocated _obj(evaluator, xs)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, xs)
    @test g == [2 * (1 - 1), 2 * (2 - 1), 2 * (3 - 1), 2 * (4 - 1)]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, xs)
    return
end

function test_objective_broadcasted_pow_cubed()
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    ArrayDiff.set_objective(model, :(sum([$x1, $x2] .^ 3)))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad])
    xs = [2.0, 3.0]
    @test MOI.eval_objective(evaluator, xs) ≈ 2.0^3 + 3.0^3
    @test 0 == @allocated _obj(evaluator, xs)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, xs)
    @test g ≈ [3 * 2.0^2, 3 * 3.0^2]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, xs)
    return
end

function test_model_typed_default_is_float64()
    model = ArrayDiff.Model()
    @test model isa ArrayDiff.Model{Float64}
    @test model.parameters isa Vector{Float64}
    @test model.expressions isa Vector{ArrayDiff.Expression{Float64}}
    @test model.constraints isa ArrayDiff.OrderedCollections.OrderedDict{
        ArrayDiff.ConstraintIndex,
        ArrayDiff.Constraint{Float64},
    }
    return
end

function test_model_typed_float32_parse_value()
    model = ArrayDiff.Model{Float32}()
    x = MOI.VariableIndex(1)
    ArrayDiff.set_objective(model, :($x + 1.5))
    obj = something(model.objective)
    @test obj isa ArrayDiff.Expression{Float32}
    @test obj.values isa Vector{Float32}
    @test obj.values == Float32[1.5]
    return
end

function test_model_typed_float32_add_parameter()
    model = ArrayDiff.Model{Float32}()
    p = ArrayDiff.add_parameter(model, 2.5)
    @test p isa ArrayDiff.ParameterIndex
    @test model.parameters isa Vector{Float32}
    @test model.parameters == Float32[2.5]
    return
end

function test_model_typed_float32_add_constraint()
    model = ArrayDiff.Model{Float32}()
    x = MOI.VariableIndex(1)
    set = MOI.LessThan{Float32}(3.0f0)
    idx = ArrayDiff.add_constraint(model, :($x + 1.0), set)
    @test idx isa ArrayDiff.ConstraintIndex
    c = model.constraints[idx]
    @test c isa ArrayDiff.Constraint{Float32}
    @test c.expression isa ArrayDiff.Expression{Float32}
    @test c.expression.values == Float32[1.0]
    @test c.set === set
    return
end

function test_model_typed_float32_add_expression()
    model = ArrayDiff.Model{Float32}()
    x = MOI.VariableIndex(1)
    idx = ArrayDiff.add_expression(model, :($x * 2.0))
    @test idx isa ArrayDiff.ExpressionIndex
    e = model[idx]
    @test e isa ArrayDiff.Expression{Float32}
    @test e.values == Float32[2.0]
    return
end

function test_model_typed_bigfloat_constraint_set()
    model = ArrayDiff.Model{BigFloat}()
    x = MOI.VariableIndex(1)
    set = MOI.GreaterThan{BigFloat}(big"1.0")
    idx = ArrayDiff.add_constraint(model, :($x), set)
    c = model.constraints[idx]
    @test c isa ArrayDiff.Constraint{BigFloat}
    @test c.set === set
    return
end

function test_model_typed_float32_evaluator_runs()
    # End-to-end smoke test: parsing happens in T = Float32, AD evaluation
    # converts to Float64 internally.
    model = ArrayDiff.Model{Float32}()
    x = MOI.VariableIndex(1)
    ArrayDiff.set_objective(model, :(2 * dot([$x], [$x]) + 1.0))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x])
    @test evaluator isa ArrayDiff.Evaluator{Float32}
    MOI.initialize(evaluator, [:Grad])
    xv = [1.5]
    @test MOI.eval_objective(evaluator, xv) ≈ 2 * xv[1]^2 + 1.0
    @test 0 == @allocated _obj(evaluator, xv)
    g = ones(1)
    MOI.eval_objective_gradient(evaluator, g, xv)
    @test g[1] ≈ 4 * xv[1]
    @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, xv)
    return
end

function test_residual_with_subexpression()
    # Residual references a subexpression `e = x1 * x2`, so the evaluator's
    # subexpression-iteration loops in `_forward_pass_residual!` and
    # `eval_residual_jprod!` are exercised.
    model = ArrayDiff.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    e = ArrayDiff.add_expression(model, :($x1 * $x2))
    # F = [x1 + e, x2 - e]
    ArrayDiff.set_residual!(model, :([$x1 + $e, $x2 - $e]))
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad, :Jac, :JacVec])
    @test ArrayDiff.residual_dimension(evaluator) == 2
    x = [3.0, 4.0]
    # e = 12, F = [3 + 12, 4 - 12] = [15, -8]
    F = zeros(2)
    ArrayDiff.eval_residual!(evaluator, F, x)
    @test F == [15.0, -8.0]
    @test 0 == @allocated ArrayDiff.eval_residual!(evaluator, F, x)
    # J = [1+x2  x1 ; -x2  1-x1] = [5 3 ; -4 -2]
    Jtv = zeros(2)
    v_ones = [1.0, 1.0]
    ArrayDiff.eval_residual_jtprod!(evaluator, Jtv, x, v_ones)
    @test Jtv == [1.0, 1.0]
    @test 0 ==
          @allocated ArrayDiff.eval_residual_jtprod!(evaluator, Jtv, x, v_ones)
    Jv = zeros(2)
    v_e1 = [1.0, 0.0]
    ArrayDiff.eval_residual_jprod!(evaluator, Jv, x, v_e1)
    @test Jv == [5.0, -4.0]
    # `eval_residual_jprod!` is not allocation-free: it allocates `seed` and
    # `row` on every call (see `src/mathoptinterface_api.jl`).
    @test_broken 0 == @allocated ArrayDiff.eval_residual_jprod!(
        evaluator,
        Jv,
        x,
        v_e1,
    )
    v_e2 = [0.0, 1.0]
    ArrayDiff.eval_residual_jprod!(evaluator, Jv, x, v_e2)
    @test Jv == [3.0, -2.0]
    @test_broken 0 == @allocated ArrayDiff.eval_residual_jprod!(
        evaluator,
        Jv,
        x,
        v_e2,
    )
    return
end

end  # module

TestArrayDiff.runtests()
