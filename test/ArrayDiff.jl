module TestArrayDiff

using Test
import LinearAlgebra
import SparseArrays

import MathOptInterface as MOI
const Nonlinear = MOI.Nonlinear

import ArrayDiff

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
    model = Nonlinear.Model()
    x = MOI.VariableIndex(1)
    Nonlinear.set_objective(model, :(dot([$x], [$x])))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x])
    MOI.initialize(evaluator, [:Grad, :Hess])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 1, 0, 1, 0]
    @test sizes.size_offset == [0, 1, 0, 0, 0]
    @test sizes.size == [1, 1]
    @test sizes.storage_offset == [0, 1, 2, 3, 4, 5]
    x = [1.2]
    @test MOI.eval_objective(evaluator, x) == x[1]^2
    g = ones(1)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g[1] == 2x[1]
    return
end

function test_objective_dot_univariate_and_scalar_mult()
    model = Nonlinear.Model()
    x = MOI.VariableIndex(1)
    Nonlinear.set_objective(model, :(2*(dot([$x], [$x]))))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 0, 0, 1, 0, 1, 0]
    @test sizes.size_offset == [0, 0, 0, 1, 0, 0, 0]
    @test sizes.size == [1, 1]
    @test sizes.storage_offset == [0, 1, 2, 3, 4, 5, 6, 7]
    x = [1.2]
    @test MOI.eval_objective(evaluator, x) == 2*x[1]^2
    g = ones(1)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g[1] == 4x[1]
    return
end

function test_objective_dot_bivariate()
    model = Nonlinear.Model()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    Nonlinear.set_objective(
        model,
        :(dot([$x, $y] - [1, 2], -[1, 2] + [$x, $y])),
    )
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x, y])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
    @test sizes.size_offset == [0, 6, 5, 0, 0, 4, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0]
    @test sizes.size == [2, 2, 2, 2, 2, 2, 2]
    @test sizes.storage_offset ==
          [0, 1, 3, 5, 6, 7, 9, 10, 11, 13, 15, 17, 18, 19, 21, 22, 23]
    x = [5, -1]
    @test MOI.eval_objective(evaluator, x) ≈ 25
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == 2(x - [1, 2])
    return
end

function test_objective_hcat_scalars()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    Nonlinear.set_objective(model, :(dot([$x1 $x3], [$x2 $x4])))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
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
    @test MOI.eval_objective(evaluator, [x1, x2, x3, x4]) == 14.0
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2, x3, x4])
    @test g == [2.0, 1.0, 4.0, 3.0]
    return
end

function test_objective_hcat_vectors()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    Nonlinear.set_objective(
        model,
        :(dot(hcat([$x1], [$x3]), hcat([$x2], [$x4]))),
    )
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
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
    @test MOI.eval_objective(evaluator, [x1, x2, x3, x4]) == 14.0
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2, x3, x4])
    @test g == [2.0, 1.0, 4.0, 3.0]
    return
end

function test_objective_dot_bivariate_on_rows()
    model = Nonlinear.Model()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    Nonlinear.set_objective(model, :(dot([$x $y] - [1 2], -[1 2] + [$x $y])))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x, y])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 2, 0, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0]
    @test sizes.size_offset ==
          [0, 12, 10, 0, 0, 8, 0, 0, 6, 4, 2, 0, 0, 0, 0, 0]
    @test sizes.size == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    @test sizes.storage_offset ==
          [0, 1, 3, 5, 6, 7, 9, 10, 11, 13, 15, 17, 18, 19, 21, 22, 23]
    x = [5, -1]
    @test MOI.eval_objective(evaluator, x) ≈ 25
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == 2(x - [1, 2])
    return
end

function test_objective_norm_univariate()
    model = Nonlinear.Model()
    x = MOI.VariableIndex(1)
    Nonlinear.set_objective(model, :(norm([$x])))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 1, 0]
    @test sizes.size_offset == [0, 0, 0]
    @test sizes.size == [1]
    @test sizes.storage_offset == [0, 1, 2, 3]
    x = [1.2]
    @test MOI.eval_objective(evaluator, x) == abs(x[1])
    g = ones(1)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g[1] == sign(x[1])
    return
end

function test_objective_norm_bivariate()
    model = Nonlinear.Model()
    x = MOI.VariableIndex(1)
    y = MOI.VariableIndex(2)
    Nonlinear.set_objective(model, :(norm([$x, $y])))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x, y])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 1, 0, 0]
    @test sizes.size_offset == [0, 0, 0, 0]
    @test sizes.size == [2]
    @test sizes.storage_offset == [0, 1, 3, 4, 5]
    x = [3.0, 4.0]
    @test MOI.eval_objective(evaluator, x) == 5.0
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, x)
    @test g == x / 5.0
    y = [0.0, 0.0]
    @test MOI.eval_objective(evaluator, y) == 0.0
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, y)
    @test g == [0.0, 0.0]
    return
end

function test_objective_norm_of_row_vector()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    Nonlinear.set_objective(model, :(norm([$x1 $x2])))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 0, 0]
    @test sizes.size_offset == [0, 0, 0, 0]
    @test sizes.size == [1, 2]
    @test sizes.storage_offset == [0, 1, 3, 4, 5]
    x1 = 1.0
    x2 = 2.0
    @test MOI.eval_objective(evaluator, [x1, x2]) == sqrt(5.0)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2])
    @test g == [1.0 / sqrt(5.0), 2.0 / sqrt(5.0)]
    return
end

function test_objective_norm_of_vcat_vector()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    Nonlinear.set_objective(model, :(norm(vcat($x1, $x3))))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
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
    @test MOI.eval_objective(evaluator, [x1, x2, x3, x4]) == sqrt(10.0)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2, x3, x4])
    @test g == [1.0 / sqrt(10.0), 0.0, 3.0 / sqrt(10.0), 0.0]
    return
end

function test_objective_norm_of_vcat_matrix()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    Nonlinear.set_objective(model, :(norm(vcat([$x1 $x3], [$x2 $x4]))))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
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
    @test MOI.eval_objective(evaluator, [x1, x2, x3, x4]) == sqrt(30.0)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2, x3, x4])
    @test g == [
        1.0 / sqrt(30.0),
        2.0 / sqrt(30.0),
        3.0 / sqrt(30.0),
        4.0 / sqrt(30.0),
    ]
    return
end

function test_objective_norm_of_row()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    Nonlinear.set_objective(model, :(norm(row($x1, $x2))))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2])
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    @test sizes.ndims == [0, 2, 0, 0]
    @test sizes.size_offset == [0, 0, 0, 0]
    @test sizes.size == [1, 2]
    @test sizes.storage_offset == [0, 1, 3, 4, 5]
    x1 = 1.0
    x2 = 2.0
    @test MOI.eval_objective(evaluator, [x1, x2]) == sqrt(5.0)
    g = ones(2)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2])
    @test g == [1.0 / sqrt(5.0), 2.0 / sqrt(5.0)]
    return
end

function test_objective_norm_of_matrix()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    Nonlinear.set_objective(model, :(norm([$x1 $x2; $x3 $x4])))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
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
    @test MOI.eval_objective(evaluator, [x1, x2, x3, x4]) == sqrt(30.0)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2, x3, x4])
    @test g == [
        1.0 / sqrt(30.0),
        2.0 / sqrt(30.0),
        3.0 / sqrt(30.0),
        4.0 / sqrt(30.0),
    ]
    return
end

function test_objective_norm_of_matrix_with_sum()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    Nonlinear.set_objective(model, :(norm([$x1 $x2; $x3 $x4] - [1 1; 1 1])))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
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
    @test MOI.eval_objective(evaluator, [x1, x2, x3, x4]) == sqrt(14.0)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2, x3, x4])
    @test g == [0.0, 1.0 / sqrt(14.0), 2.0 / sqrt(14.0), 3.0 / sqrt(14.0)]
    return
end

function test_objective_norm_of_product_of_matrices()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    Nonlinear.set_objective(model, :(norm([$x1 $x2; $x3 $x4] * [1 0; 0 1])))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
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
    @test MOI.eval_objective(evaluator, [x1, x2, x3, x4]) == sqrt(30.0)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2, x3, x4])
    @test g == [
        1.0 / sqrt(30.0),
        2.0 / sqrt(30.0),
        3.0 / sqrt(30.0),
        4.0 / sqrt(30.0),
    ]
    return
end

function test_objective_norm_of_product_of_matrices_with_sum()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    Nonlinear.set_objective(
        model,
        :(norm(([$x1 $x2; $x3 $x4] + [1 1; 1 1]) * [1 0; 0 1])),
    )
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
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
    @test MOI.eval_objective(evaluator, [x1, x2, x3, x4]) == sqrt(54.0)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2, x3, x4])
    @test g == [
        2.0 / sqrt(54.0),
        3.0 / sqrt(54.0),
        4.0 / sqrt(54.0),
        5.0 / sqrt(54.0),
    ]
    return
end

function test_objective_norm_of_mtx_vector_product()
    model = Nonlinear.Model()
    x1 = MOI.VariableIndex(1)
    x2 = MOI.VariableIndex(2)
    x3 = MOI.VariableIndex(3)
    x4 = MOI.VariableIndex(4)
    Nonlinear.set_objective(model, :(norm(([$x1 $x2; $x3 $x4] * [1; 1]))))
    evaluator = Nonlinear.Evaluator(model, ArrayDiff.Mode(), [x1, x2, x3, x4])
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
    @test MOI.eval_objective(evaluator, [x1, x2, x3, x4]) == sqrt(58.0)
    g = ones(4)
    MOI.eval_objective_gradient(evaluator, g, [x1, x2, x3, x4])
    @test g == [
        3.0 / sqrt(58.0),
        3.0 / sqrt(58.0),
        7.0 / sqrt(58.0),
        7.0 / sqrt(58.0),
    ]
    return
end

end  # module

TestArrayDiff.runtests()
