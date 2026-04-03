module TestJuMP

using Test

using JuMP
using ArrayDiff
import LinearAlgebra

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

function test_l2_loss_simple()
    n = 2
    X = rand(n, n)
    Y = rand(n, n)
    model = Model()
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    prod = W2 * X
    diff_expr = prod .- Y
    @test diff_expr isa ArrayDiff.MatrixExpr
    return
end

function test_l2_loss_tanh()
    n = 2
    X = rand(n, n)
    Y = rand(n, n)
    model = Model()
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    hidden = tanh.(W1 * X)
    diff_expr = hidden .- Y
    @test diff_expr isa ArrayDiff.MatrixExpr
    return
end

end  # module

TestJuMP.runtests()
