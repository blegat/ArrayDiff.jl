module TestJuMP

using Test

using JuMP
using ArrayDiff
import LinearAlgebra
import MathOptInterface as MOI
import NLopt

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

function test_neural()
    n = 2
    X = rand(n, n)
    model = Model()
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @test W1 isa ArrayDiff.MatrixOfVariables{Float64}
    @test JuMP.index(W1[1, 1]) == MOI.VariableIndex(1)
    @test JuMP.index(W1[2, 1]) == MOI.VariableIndex(2)
    @test JuMP.index(W1[2]) == MOI.VariableIndex(2)
    @test sprint(show, W1) ==
          "2×2 ArrayDiff.ArrayOfVariables{Float64, 2} with offset 0"
    for prod in [W1 * X, X * W1]
        @test prod isa ArrayDiff.MatrixExpr
        @test prod.head == :*
        @test !prod.broadcasted
        @test sprint(show, prod) ==
              "2×2 ArrayDiff.GenericArrayExpr{JuMP.VariableRef, 2}"
        err = ErrorException(
            "`getindex` not implemented, build vectorized expression instead",
        )
        @test_throws err prod[1, 1]
    end
    Y1 = W1 * X
    X1 = tanh.(Y1)
    @test X1 isa ArrayDiff.MatrixExpr
    @test X1.head == :tanh
    @test X1.broadcasted
    @test X1.args[] === Y1
    Y2 = W2 * X1
    @test Y2.head == :*
    @test !Y2.broadcasted
    @test length(Y2.args) == 2
    @test Y2.args[1] === W2
    @test Y2.args[2] === X1
    return
end

function test_binary_broadcasting()
    n = 2
    model = Model()
    @variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    Y = rand(n, n)
    D1 = W .- Y
    @test D1 isa ArrayDiff.MatrixExpr
    @test D1.head == :-
    @test D1.broadcasted
    @test size(D1) == (n, n)
    @test D1.args[1] === W
    @test D1.args[2] === Y
    D2 = Y .- W
    @test D2 isa ArrayDiff.MatrixExpr
    @test D2.head == :-
    @test D2.broadcasted
    @test D2.args[1] === Y
    @test D2.args[2] === W
    @variable(model, V[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    D3 = W .- V
    @test D3 isa ArrayDiff.MatrixExpr
    @test D3.head == :-
    @test D3.broadcasted
    @test D3.args[1] === W
    @test D3.args[2] === V
    return
end

function test_norm()
    n = 2
    model = Model()
    @variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    loss = LinearAlgebra.norm(W)
    @test loss isa JuMP.NonlinearExpr
    @test loss.head == :norm
    @test length(loss.args) == 1
    @test loss.args[1] === W
    return
end

function test_l2_loss()
    n = 2
    X = rand(n, n)
    Y = rand(n, n)
    model = Model()
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    Y_hat = W2 * tanh.(W1 * X)
    diff_expr = Y_hat .- Y
    @test diff_expr isa ArrayDiff.MatrixExpr
    @test diff_expr.head == :-
    @test diff_expr.broadcasted
    @test diff_expr.args[1] === Y_hat
    @test diff_expr.args[2] === Y
    loss = LinearAlgebra.norm(diff_expr)
    @test loss isa JuMP.NonlinearExpr
    @test loss.head == :norm
    @test loss.args[1] === diff_expr
end

function test_array_subtraction()
    model = Model()
    @variable(model, W[1:2, 1:2], container = ArrayDiff.ArrayOfVariables)
    X = rand(2, 2)
    diff = W * X - X
    @test diff isa ArrayDiff.MatrixExpr
    @test diff.head == :-
    @test size(diff) == (2, 2)
    return
end

function test_array_addition()
    model = Model()
    @variable(model, W[1:2, 1:2], container = ArrayDiff.ArrayOfVariables)
    X = rand(2, 2)
    s = W * X + X
    @test s isa ArrayDiff.MatrixExpr
    @test s.head == :+
    @test size(s) == (2, 2)
    return
end

function test_to_expr()
    model = Model()
    @variable(model, W[1:2, 1:2], container = ArrayDiff.ArrayOfVariables)
    X = rand(2, 2)
    Y = W * tanh.(W * X)
    diff = Y - X
    loss = ArrayDiff.sumsq(diff)
    expr = ArrayDiff.to_expr(loss)
    @test expr isa Expr
    @test expr.head == :call
    @test expr.args[1] == :dot
    return
end

function test_moi_function()
    model = Model()
    @variable(model, W[1:2, 1:2], container = ArrayDiff.ArrayOfVariables)
    X = rand(2, 2)
    Y = W * X
    f = JuMP.moi_function(Y)
    @test f isa ArrayDiff.ArrayNonlinearFunction{2}
    @test f.head == :*
    @test f.size == (2, 2)
    @test !f.broadcasted
    @test MOI.output_dimension(f) == 4
    return
end

function test_neural_nlopt()
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    model = Model(NLopt.Optimizer)
    set_attribute(model, "algorithm", :LD_LBFGS)
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    # Use distinct starting values to break symmetry
    start_W1 = [0.3 -0.2; 0.1 0.4]
    start_W2 = [-0.1 0.5; 0.2 -0.3]
    for i in 1:n, j in 1:n
        set_start_value(W1[i, j], start_W1[i, j])
        set_start_value(W2[i, j], start_W2[i, j])
    end
    Y = W2 * tanh.(W1 * X)
    diff = Y - target
    loss = ArrayDiff.sumsq(diff)
    ArrayDiff.set_nlp_objective!(model, MOI.MIN_SENSE, loss)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) < 1e-6
    return
end

end  # module

TestJuMP.runtests()
