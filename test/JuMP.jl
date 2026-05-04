module TestJuMP

using Test

using JuMP
using ArrayDiff
import LinearAlgebra
import MathOptInterface as MOI

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
              "2×2 ArrayDiff.GenericArrayExpr{$(JuMP.VariableRef), 2}"
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

function test_parse_moi()
    # Test that ArrayDiff.Model can parse ScalarNonlinearFunction
    # with ArrayNonlinearFunction args
    model = Model()
    @variable(model, W[1:2, 1:2], container = ArrayDiff.ArrayOfVariables)
    X = rand(2, 2)
    Y = W * X
    diff = Y .- X
    loss = LinearAlgebra.norm(diff)
    snf = JuMP.moi_function(loss)
    @test snf isa MOI.ScalarNonlinearFunction
    @test snf.head == :norm
    @test snf.args[] isa ArrayDiff.ArrayNonlinearFunction{2}
    ad_model = ArrayDiff.Model()
    ArrayDiff.set_objective(ad_model, snf)
    @test ad_model.objective !== nothing
    loss = sum(diff .^ 2)
    snf = JuMP.moi_function(loss)
    @test snf isa MOI.ScalarNonlinearFunction
    @test snf.head == :sum
    next = snf.args[]
    @test next isa ArrayDiff.ArrayNonlinearFunction{2}
    @test next.head == :^
    return
end

function _eval(model, func, x)
    mode = ArrayDiff.Mode()
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(func))
    evaluator = MOI.Nonlinear.Evaluator(ad, mode, JuMP.index.(JuMP.all_variables(model)))
    MOI.initialize(evaluator, [:Grad])
    val = MOI.eval_objective(evaluator, x)
    g = zero(x)
    MOI.eval_objective_gradient(evaluator, g, Float64.(collect(1:8)))
    MOI.Nonlinear.set_objective(ad, nothing)
    @test isnothing(ad.objective)
    return val, g
end

function _test_neural(with_norm::Bool, broadcast::Bool, plus::Bool)
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    if plus
        target = -target
    end
    model = Model()
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    # Use distinct starting values to break symmetry
    Y = W2 * tanh.(W1 * X)
    if plus
        if broadcast
            E = Y .+ target
        else
            E = Y + target
        end
    else
        if broadcast
            E = Y .- target
        else
            E = Y - target
        end
    end
    if with_norm
        loss = LinearAlgebra.norm(E)
    else
        loss = sum(E .^ 2)
    end
    W1_val = [0.3 -0.2; 0.1 0.4]
    W2_val = [-0.1 0.5; 0.2 -0.3]
    obj, g = _eval(model, loss, [vec(W1_val); vec(W2_val)])
    obj_val = 0.8516435891643307
    if with_norm
        obj_val = sqrt(obj_val)
    end
    @test obj ≈ obj_val
    grad = [
        12.3913945850742
        0.6880048864793
        9.4322503589489
        0.5223651220724
        46.2269560438734
        53.9729454980064
        45.7401048264386
        53.4195902684781
    ]
    if with_norm
        @test g ≈ grad * 0.019879429552408144
    else
        @test g ≈ grad
    end
    return
end

function test_neural()
    bin = [false, true]
    @testset "$(with_norm ? "norm" : "sum")" for with_norm in bin
        @testset "$(broadcast ? "broadcast" : "array")" for broadcast in bin
            @testset "$(plus ? "+" : "-")" for plus in bin
                _test_neural(with_norm, broadcast, plus)
            end
        end
    end
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

end  # module

TestJuMP.runtests()
