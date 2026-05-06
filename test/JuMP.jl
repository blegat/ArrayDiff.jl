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

function _eval(model::JuMP.GenericModel{T}, func, x) where {T}
    mode = ArrayDiff.Mode{Vector{T}}()
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(func))
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    val = MOI.eval_objective(evaluator, x)
    g = zero(x)
    MOI.eval_objective_gradient(evaluator, g, T.(collect(1:8)))
    MOI.Nonlinear.set_objective(ad, nothing)
    @test isnothing(ad.objective)
    return val, g
end

function _test_neural(
    with_norm::Bool,
    broadcast::Bool,
    plus::Bool,
    wrap::Bool,
    swap::Bool,
    T::Type,
)
    n = 2
    X = T[1.0 0.5; 0.3 0.8]
    target = T[0.5 0.2; 0.1 0.7]
    if wrap
        ME = ArrayDiff.GenericMatrixExpr{JuMP.GenericVariableRef{T}}
        X = ME(:+, Any[X], size(X), false)
        target = ME(:+, Any[target], size(target), false)
    end
    if plus
        target = -target
    end
    model = GenericModel{T}()
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    # Use distinct starting values to break symmetry
    Y = W2 * tanh.(W1 * X)
    if plus
        if broadcast
            if swap
                E = target .+ Y
            else
                E = Y .+ target
            end
        else
            if swap
                E = target + Y
            else
                E = Y + target
            end
        end
    else
        if broadcast
            if swap
                E = target .- Y
            else
                E = Y .- target
            end
        else
            if swap
                E = target - Y
            else
                E = Y - target
            end
        end
    end
    if with_norm
        loss = LinearAlgebra.norm(E)
    else
        loss = sum(E .^ 2)
    end
    W1_val = T[0.3 -0.2; 0.1 0.4]
    W2_val = T[-0.1 0.5; 0.2 -0.3]
    obj, g = _eval(model, loss, [vec(W1_val); vec(W2_val)])
    # Reference computed from the same hand-written forward/reverse formulas
    # as `perf/cuda_vs_pytorch.jl::forward_pass`/`reverse_diff`, adapted to
    # this test's loss `sum((Y - target).^2)` (no `/ n` scaling, full gradient
    # over both `W1` and `W2`). `_eval` evaluates the objective at `xstart`
    # and the gradient at `x = [1, ..., 8]`, so we need the references at the
    # corresponding inputs.
    X_const = T[1.0 0.5; 0.3 0.8]
    target_const = T[0.5 0.2; 0.1 0.7]
    obj_val = _ref_objective(W1_val, W2_val, X_const, target_const)
    if with_norm
        obj_val = sqrt(obj_val)
    end
    @test obj ≈ obj_val
    W1_at_grad = reshape(T[1.0, 2.0, 3.0, 4.0], 2, 2)
    W2_at_grad = reshape(T[5.0, 6.0, 7.0, 8.0], 2, 2)
    grad_sumsq = _ref_gradient(W1_at_grad, W2_at_grad, X_const, target_const)
    if with_norm
        # `d/dx ‖E‖₂ = (1/(2‖E‖₂)) · d/dx ‖E‖₂² = grad_sumsq / (2 sqrt(sumsq))`,
        # taken at the gradient evaluation point.
        norm_at_grad =
            sqrt(_ref_objective(W1_at_grad, W2_at_grad, X_const, target_const))
        @test g ≈ grad_sumsq ./ (2 * norm_at_grad)
    else
        @test g ≈ grad_sumsq
    end
    return
end

# Hand-written forward + reverse for the 2-layer MLP `loss = sum((W2 *
# tanh.(W1 * X) - target).^2)`. Same shape as `perf/cuda_vs_pytorch.jl`'s
# `forward_pass` / `reverse_diff` but adapted to this test (no `/ n` scaling
# and gradient over both `W1` and `W2`). Returned gradient is flattened with
# the JuMP variable convention `[vec(grad_W1); vec(grad_W2)]`.
function _ref_forward(W1, W2, X, target)
    y_1 = tanh.(W1 * X)
    J_1 = 1 .- y_1 .^ 2
    J_2 = 2 .* (W2 * y_1 .- target)
    return y_1, J_1, J_2
end

function _ref_objective(W1, W2, X, target)
    return sum((W2 * tanh.(W1 * X) .- target) .^ 2)
end

function _ref_gradient(W1, W2, X, target)
    y_1, J_1, J_2 = _ref_forward(W1, W2, X, target)
    grad_W1 = (J_1 .* (W2' * J_2)) * X'
    grad_W2 = J_2 * y_1'
    return [vec(grad_W1); vec(grad_W2)]
end

function test_neural()
    bin = [false, true]
    @testset "$(with_norm ? "norm" : "sum")" for with_norm in bin
        @testset "$(broadcast ? "broadcast" : "array")" for broadcast in bin
            @testset "$(plus ? "+" : "-")" for plus in bin
                @testset "$(wrap ? "wrap" : "nowrap")" for wrap in bin
                    @testset "$(swap ? "swap" : "noswap")" for swap in bin
                        @testset "$T" for T in [Float64, Float32]
                            _test_neural(
                                with_norm,
                                broadcast,
                                plus,
                                wrap,
                                swap,
                                T,
                            )
                        end
                    end
                end
            end
        end
    end
end

# Builds the same `sum((W2*tanh.(W1*X) - target)^2)` MLP that `test_neural`
# exercises and checks that, after warmup, both `eval_objective` and
# `eval_objective_gradient` are allocation-free on the CPU `Vector{Float64}`
# tape — including when the input `x` has changed since the last call (which
# is the path that actually re-runs forward+reverse, not the
# `last_x == x` short-circuit).
function test_neural_allocations()
    if VERSION < v"1.12"
        return
    end
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    model = Model()
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    Y = W2 * tanh.(W1 * X)
    loss = sum((Y .- target) .^ 2)
    mode = ArrayDiff.Mode()
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(loss))
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    x1 = Float64.(collect(1:8))
    x2 = Float64.(collect(2:9))
    g = zeros(8)
    # Wrapped in typed functions so `@allocated` doesn't capture the
    # return-value boxing that happens when calling `eval_objective`
    # directly from the macro's untyped scope (each `MOI.eval_objective`
    # returns a `Float64` which then escapes into `Any`-typed scope).
    _obj(ev, x) = MOI.eval_objective(ev, x)
    function _grad!(ev, g, x)
        MOI.eval_objective_gradient(ev, g, x)
        return nothing
    end
    # Warmup: trigger JIT compilation for both `eval_objective` and
    # `eval_objective_gradient`. Two distinct inputs so `_reverse_mode`'s
    # `last_x == x` short-circuit doesn't elide the work on the second call.
    _obj(evaluator, x1)
    _obj(evaluator, x2)
    _grad!(evaluator, g, x1)
    _grad!(evaluator, g, x2)
    # Now alternate: each measured call sees `last_x ≠ x`, so it actually
    # runs the full forward + reverse passes through the block tape.
    @test 0 == @allocated _obj(evaluator, x1)
    @test 0 == @allocated _obj(evaluator, x2)
    @test 0 == @allocated _grad!(evaluator, g, x1)
    @test 0 == @allocated _grad!(evaluator, g, x2)
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

end  # module

TestJuMP.runtests()
