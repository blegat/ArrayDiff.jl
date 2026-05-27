module TestJuMP

using Test

using JuMP
using ArrayDiff
import ChainRulesCore
import LinearAlgebra
import MathOptInterface as MOI

include(joinpath(@__DIR__, "Transformer.jl"))

# Helpers used by the ChainRules tests below. Defined at module scope so
# `Symbol(my_relu)` resolves to `:my_relu` (rather than a generated closure
# name) and so `ChainRulesCore.rrule` can be overloaded for `my_crossentropy`.
my_relu(x) = max(zero(x), x)
# No need to define any ChainRules because FiniteDiff is used in the broadcast.

function my_crossentropy(p, q)
    return -sum(q .* log.(p .+ 1e-3))
end

function ChainRulesCore.rrule(
    ::typeof(my_crossentropy),
    p::AbstractArray,
    q::AbstractArray,
)
    ε = 1e-3
    val = my_crossentropy(p, q)
    function pullback(δ)
        dp = δ .* (-q ./ (p .+ ε))
        dq = δ .* (-log.(p .+ ε))
        return ChainRulesCore.NoTangent(), dp, dq
    end
    return val, pullback
end

# `my_crossentropy1` and `my_crossentropy2` exercise the `infer_sizes` API:
# the first relies on the default zeros-probe, the second has an explicit
# override. To prove the override is what runs, `my_crossentropy2` asserts
# `all(p .> 0)` — the default probe would build `p = zeros(...)` and assert-
# fail, so reaching the test's final `@test` confirms `infer_sizes` was
# specialised.
my_crossentropy1(p, q) = -sum(q .* log.(p .+ 1e-3))
function my_crossentropy2(p, q)
    @assert all(>(0), p)
    return -sum(q .* log.(p))
end

function ArrayDiff.infer_sizes(::typeof(my_crossentropy2), ::Tuple, ::Tuple)
    return ()
end

function ChainRulesCore.rrule(
    ::typeof(my_crossentropy1),
    p::AbstractArray,
    q::AbstractArray,
)
    ε = 1e-3
    val = my_crossentropy1(p, q)
    function pullback(δ)
        dp = δ .* (-q ./ (p .+ ε))
        dq = δ .* (-log.(p .+ ε))
        return ChainRulesCore.NoTangent(), dp, dq
    end
    return val, pullback
end

function ChainRulesCore.rrule(
    ::typeof(my_crossentropy2),
    p::AbstractArray,
    q::AbstractArray,
)
    val = my_crossentropy2(p, q)
    function pullback(δ)
        dp = δ .* (-q ./ p)
        dq = δ .* (-log.(p))
        return ChainRulesCore.NoTangent(), dp, dq
    end
    return val, pullback
end

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

function _eval(
    model::JuMP.GenericModel{T},
    func,
    x;
    x_grad = T.(collect(1:length(x))),
) where {T}
    mode = ArrayDiff.Mode{Vector{T}}()
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(func))
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    sizes = evaluator.backend.objective.expr.sizes
    val = MOI.eval_objective(evaluator, x)
    if VERSION >= v"1.12"
        fill!(evaluator.backend.last_x, NaN)
        @test 0 == @allocated MOI.eval_objective(evaluator, x)
    end
    g = zero(x)
    MOI.eval_objective_gradient(evaluator, g, x_grad)
    if VERSION >= v"1.12"
        fill!(evaluator.backend.last_x, NaN)
        @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x_grad)
    end
    MOI.Nonlinear.set_objective(ad, nothing)
    @test isnothing(ad.objective)
    return sizes, val, g, evaluator
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
    W1_at_grad = reshape(T[1.0, 2.0, 3.0, 4.0], 2, 2)
    W2_at_grad = reshape(T[5.0, 6.0, 7.0, 8.0], 2, 2)
    grad_sumsq = _ref_gradient(W1_at_grad, W2_at_grad, X_const, target_const)
    if with_norm
        # `d/dx ‖E‖₂ = (1/(2‖E‖₂)) · d/dx ‖E‖₂² = grad_sumsq / (2 sqrt(sumsq))`,
        # taken at the gradient evaluation point.
        norm_at_grad =
            sqrt(_ref_objective(W1_at_grad, W2_at_grad, X_const, target_const))
        grad_val = grad_sumsq ./ (2 * norm_at_grad)
    else
        grad_val = grad_sumsq
    end
    _, val, g = _eval(model, loss, [vec(W1_val); vec(W2_val)])
    @test obj_val ≈ val
    @test grad_val ≈ g
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

# Build the non-broadcasted `:*` size-inference cases the HEAD commit fixed.
# JuMP's surface syntax always lowers `c * W` to a broadcasted node, so to
# exercise the non-broadcasted code path we build the `MatrixExpr` directly
# (same pattern `_test_neural` uses for `wrap`).
function test_size_inference_scalar_times_matrix()
    mode = ArrayDiff.Mode()
    ME = ArrayDiff.GenericMatrixExpr{VariableRef}
    @testset "$(rows)x$(cols)" for (rows, cols) in [(2, 3), (3, 2), (2, 2)]
        model = Model()
        @variable(
            model,
            W[1:rows, 1:cols],
            container = ArrayDiff.ArrayOfVariables,
        )
        @testset "$(name)" for (name, expr) in [
            ("scalar * M", ME(:*, Any[2.5, W], (rows, cols), false)),
            ("M * scalar", ME(:*, Any[W, 2.5], (rows, cols), false)),
        ]
            ad = ArrayDiff.model(mode)
            MOI.Nonlinear.set_objective(
                ad,
                JuMP.moi_function(LinearAlgebra.norm(expr)),
            )
            evaluator = MOI.Nonlinear.Evaluator(
                ad,
                mode,
                JuMP.index.(JuMP.all_variables(model)),
            )
            MOI.initialize(evaluator, [:Grad])
            sizes = evaluator.backend.objective.expr.sizes
            # Tape: norm (k=1, scalar), * (k=2, matrix), then the scalar leaf
            # and the matrix leaf in some order. The * node must inherit the
            # (rows, cols) shape from the matrix child.
            @test sizes.ndims[1] == 0
            @test sizes.ndims[2] == 2
            mul_off = sizes.size_offset[2]
            @test sizes.size[mul_off+1] == rows
            @test sizes.size[mul_off+2] == cols
            # Storage for the * node should be `rows * cols`, not `1` (which
            # is what the old `(1, 1)` stub produced).
            @test sizes.storage_offset[3] - sizes.storage_offset[2] ==
                  rows * cols
            # Exactly one of the two children is the scalar leaf.
            @test sort(sizes.ndims[3:4]) == [0, 2]
            # Two ndims=2 nodes (the * and the matrix leaf) each contribute
            # a (rows, cols) entry to the flat size vector.
            @test sort(sizes.size) == sort([rows, cols, rows, cols])
        end
    end
    return
end

function test_size_vec_vect()
    mode = ArrayDiff.Mode()
    ME = ArrayDiff.GenericMatrixExpr{VariableRef}
    @testset "$(rows)x$(cols)" for (rows, cols) in [(2, 3), (3, 2), (2, 2)]
        model = Model()
        @variable(model, a[1:rows], container = ArrayDiff.ArrayOfVariables,)
        b = ones(cols)
        ad = ArrayDiff.model(mode)
        # a * b' is redirected to broadcast(*, a, b') but we want to test product here
        # this calls reshape(a, length(a), 1)
        expr = a * Matrix(b')
        MOI.Nonlinear.set_objective(ad, JuMP.moi_function(sum(expr)))
        evaluator = MOI.Nonlinear.Evaluator(
            ad,
            mode,
            JuMP.index.(JuMP.all_variables(model)),
        )
        MOI.initialize(evaluator, [:Grad])
        sizes = evaluator.backend.objective.expr.sizes
        # Tape: norm (k=1, scalar), * (k=2, matrix), then the scalar leaf
        # and the matrix leaf in some order. The * node must inherit the
        # (rows, cols) shape from the matrix child.
        @test sizes.ndims[1] == 0
        @test sizes.ndims[2] == 2
        mul_off = sizes.size_offset[2]
        @test sizes.size[mul_off+1] == rows
        @test sizes.size[mul_off+2] == cols
    end
    return
end

function test_broadcast_nonsquare_matrix()
    model = Model()
    @variable(model, W[1:2, 1:3], container = ArrayDiff.ArrayOfVariables)
    Y = [10.0 20.0 30.0; 40.0 50.0 60.0]
    x = Float64.(collect(1:6))
    W_val = reshape(x, 2, 3)
    @testset "$(op)" for (op, expr, ref_mat) in [
        (:+, LinearAlgebra.norm(W .+ Y), W_val .+ Y),
        (:-, LinearAlgebra.norm(W .- Y), W_val .- Y),
        (:*, LinearAlgebra.norm(W .* W), W_val .* W_val),
    ]
        sizes, val, g = _eval(model, expr, x)
        # Outer norm scalar, then the broadcasted op produces a 2x3 matrix,
        # then the two 2x3 leaves: 4 nodes, three of them ndims=2 with size
        # (2, 3). The old bug would report (2, 2) for the broadcast node.
        @test sizes.ndims == [0, 2, 2, 2]
        @test sizes.size == [2, 3, 2, 3, 2, 3]
        @test sizes.size_offset == [0, 4, 2, 0]
        @test sizes.storage_offset == [0, 1, 7, 13, 19]
        @test val ≈ LinearAlgebra.norm(ref_mat)
        ref_g = if op == :+
            vec(W_val .+ Y) ./ LinearAlgebra.norm(ref_mat)
        elseif op == :-
            vec(W_val .- Y) ./ LinearAlgebra.norm(ref_mat)
        else  # :*
            # d(norm(W .* W))/dW = 2 .* W .^ 3 / norm(W .* W)
            vec(2 .* W_val .^ 3) ./ LinearAlgebra.norm(ref_mat)
        end
        @test g ≈ ref_g
    end
    return
end

# Cover every `Number op MatrixVar` / `MatrixVar op Number` broadcast
# pattern that JuMP's `Base.broadcasted` produces — both the size inference
# (broadcast node inherits the matrix child's shape, not the old `(1, 1)`
# stub) and the eval/reverse paths (`out .= s op v`, `rev_s =
# ±sum(rev_parent)` or `dot(rev_parent, v)`). Loss is `norm(c op W)` so the
# analytic gradient is `dexpr_dW .* (c op W) ./ norm(c op W)`.
function test_broadcast_scalar_matrix_gradient()
    c = 2.5
    rows, cols = 2, 3
    model = Model()
    @variable(model, W[1:rows, 1:cols], container = ArrayDiff.ArrayOfVariables)
    x = Float64.(collect(1:(rows*cols)))
    W_val = reshape(x, rows, cols)
    @testset "$(name)" for (name, expr, ref_mat, dexpr_dW) in [
        ("scalar .+ M", c .+ W, c .+ W_val, fill(1.0, rows, cols)),
        ("M .+ scalar", W .+ c, W_val .+ c, fill(1.0, rows, cols)),
        ("scalar .- M", c .- W, c .- W_val, fill(-1.0, rows, cols)),
        ("M .- scalar", W .- c, W_val .- c, fill(1.0, rows, cols)),
        ("scalar .* M", c .* W, c .* W_val, fill(c, rows, cols)),
        ("M .* scalar", W .* c, W_val .* c, fill(c, rows, cols)),
    ]
        sizes, val, g = _eval(model, LinearAlgebra.norm(expr), x)
        # Outer norm scalar (k=1), then the broadcast (k=2) which must
        # inherit the matrix child's (rows, cols) shape — not the old
        # `(1, 1)` stub — then the two children (one scalar leaf, one
        # matrix leaf) in some order.
        @test sizes.ndims[1] == 0
        @test sizes.ndims[2] == 2
        b_off = sizes.size_offset[2]
        @test sizes.size[b_off+1] == rows
        @test sizes.size[b_off+2] == cols
        @test 0 in sizes.ndims[3:4]
        @test val ≈ LinearAlgebra.norm(ref_mat)
        @test g ≈ vec(dexpr_dW .* ref_mat) ./ LinearAlgebra.norm(ref_mat)
    end
    return
end

# Cover broadcasting where one operand is a column vector or a row vector
# (vector-transpose) and the other is the matrix variable W. Same loss shape
# as `test_broadcast_scalar_matrix_gradient` — `norm(c op W)` — so the
# analytic gradient is `dexpr_dW .* (c op W) ./ norm(c op W)`.
function test_broadcast_vector_matrix_gradient()
    rows, cols = 2, 3
    model = Model()
    @variable(model, W[1:rows, 1:cols], container = ArrayDiff.ArrayOfVariables)
    v = [10.0, 20.0]                  # length-rows column vector
    r = [100.0 200.0 300.0]           # 1×cols row vector (vector-transpose)
    x = Float64.(collect(1:(rows*cols)))
    W_val = reshape(x, rows, cols)
    # Broadcast partials: `dexpr_dW` is the elementwise ∂(c op W)/∂W,
    # broadcast to W's (rows, cols) shape.
    v_bcast = v .* ones(rows, cols)   # repeats v across cols
    r_bcast = ones(rows) .* r         # repeats r down rows
    @testset "$(name)" for (name, expr, ref_mat, dexpr_dW) in [
        # Column-vector broadcast (v repeats across cols)
        ("v .+ W", v .+ W, v .+ W_val, fill(1.0, rows, cols)),
        ("W .+ v", W .+ v, W_val .+ v, fill(1.0, rows, cols)),
        ("v .- W", v .- W, v .- W_val, fill(-1.0, rows, cols)),
        ("W .- v", W .- v, W_val .- v, fill(1.0, rows, cols)),
        ("v .* W", v .* W, v .* W_val, v_bcast),
        ("W .* v", W .* v, W_val .* v, v_bcast),
        # Row-vector broadcast (r repeats down rows)
        ("r .+ W", r .+ W, r .+ W_val, fill(1.0, rows, cols)),
        ("W .+ r", W .+ r, W_val .+ r, fill(1.0, rows, cols)),
        ("r .- W", r .- W, r .- W_val, fill(-1.0, rows, cols)),
        ("W .- r", W .- r, W_val .- r, fill(1.0, rows, cols)),
        ("r .* W", r .* W, r .* W_val, r_bcast),
        ("W .* r", W .* r, W_val .* r, r_bcast),
    ]
        sizes, val, g = _eval(model, LinearAlgebra.norm(expr), x)
        # Tape: norm (k=1, scalar) then the broadcast (k=2, matrix) inheriting
        # (rows, cols) from the result shape — not from the smaller operand.
        @test sizes.ndims[1] == 0
        @test sizes.ndims[2] == 2
        b_off = sizes.size_offset[2]
        @test sizes.size[b_off+1] == rows
        @test sizes.size[b_off+2] == cols
        @test val ≈ LinearAlgebra.norm(ref_mat)
        @test g ≈ vec(dexpr_dW .* ref_mat) ./ LinearAlgebra.norm(ref_mat)
    end
    return
end

# Outer-product-shape broadcast: a vector variable `v` (length rows) combined
# with a row-vector constant `r` (1×cols). The result is rows×cols, and the
# gradient w.r.t. v reduces along the broadcasted (cols) dimension.
function test_broadcast_outer_vector_gradient()
    rows, cols = 2, 3
    model = Model()
    @variable(model, v[1:rows], container = ArrayDiff.ArrayOfVariables)
    r = [100.0 200.0 300.0]           # 1×cols row vector
    x = Float64.(collect(1:rows))
    v_val = copy(x)
    r_bcast = ones(rows) .* r         # ∂(v .* r)/∂v broadcast to (rows, cols)
    @testset "$(name)" for (name, expr, ref_mat, dexpr_dv) in [
        ("v .+ r", v .+ r, v_val .+ r, fill(1.0, rows, cols)),
        ("r .+ v", r .+ v, r .+ v_val, fill(1.0, rows, cols)),
        ("v .- r", v .- r, v_val .- r, fill(1.0, rows, cols)),
        ("r .- v", r .- v, r .- v_val, fill(-1.0, rows, cols)),
        ("v .* r", v .* r, v_val .* r, r_bcast),
        ("r .* v", r .* v, r .* v_val, r_bcast),
    ]
        sizes, val, g = _eval(model, LinearAlgebra.norm(expr), x)
        @test sizes.ndims[1] == 0
        @test sizes.ndims[2] == 2
        b_off = sizes.size_offset[2]
        @test sizes.size[b_off+1] == rows
        @test sizes.size[b_off+2] == cols
        @test val ≈ LinearAlgebra.norm(ref_mat)
        # `d norm(M) / d v_i = sum_j dexpr_dv[i,j] * M[i,j] / norm(M)` because
        # v's column is broadcast across every output column.
        @test g ≈
              vec(sum(dexpr_dv .* ref_mat; dims = 2)) ./
              LinearAlgebra.norm(ref_mat)
    end
    return
end

# Plug JuMP variable matrices into the Transformer's `MLP` building block
# (`gelu(x * c_fc) * c_proj`) and confirm the forward+reverse pass runs
# end-to-end through the ArrayDiff evaluator. `gelu` exercises every
# scalar-broadcast pattern that ArrayDiff supports for `MatrixExpr`:
# `Number * matrix` scaling, `Number .* matrix`, and `Number .+ matrix`.
# We finite-difference the analytic gradient as a sanity check.
function test_transformer_mlp_gradient()
    d_emb, d_hidden, seq = 2, 3, 2
    model = Model()
    @variable(
        model,
        c_fc[1:d_emb, 1:d_hidden],
        container = ArrayDiff.ArrayOfVariables,
    )
    @variable(
        model,
        c_proj[1:d_hidden, 1:d_emb],
        container = ArrayDiff.ArrayOfVariables,
    )
    mlp = MLP(c_fc, c_proj)
    x = rand(seq, d_emb)
    loss = sum(mlp(x) .^ 2)
    mode = ArrayDiff.Mode()
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(loss))
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    nvar = JuMP.num_variables(model)
    @test nvar == 2 * d_emb * d_hidden
    x_pt = randn(nvar)
    val = MOI.eval_objective(evaluator, x_pt)
    @test isfinite(val)
    @test val >= 0
    g = zeros(nvar)
    MOI.eval_objective_gradient(evaluator, g, x_pt)
    @test all(isfinite, g)
    @test !all(iszero, g)
    # Central finite differences on the AD-built objective.
    h = 1e-6
    g_fd = zeros(nvar)
    for i in 1:nvar
        xp = copy(x_pt)
        xp[i] += h
        xm = copy(x_pt)
        xm[i] -= h
        g_fd[i] =
            (
                MOI.eval_objective(evaluator, xp) -
                MOI.eval_objective(evaluator, xm)
            ) / (2h)
    end
    @test isapprox(g, g_fd; rtol = 1e-4)
    return
end

function test_chainrules_crossentropy_of_relu()
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    model = Model()
    @variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    mode = ArrayDiff.Mode()
    ad = ArrayDiff.model(mode)
    MOI.set(
        ad,
        ArrayDiff.UserDefinedArrayOperator(:my_relu; arity = 1),
        my_relu,
    )
    MOI.set(
        ad,
        ArrayDiff.UserDefinedArrayOperator(:my_crossentropy; arity = 2),
        my_crossentropy,
    )
    Y = W * X
    Z = my_relu.(Y)
    # JuMP's `GenericNonlinearExpr` rejects raw array arguments, so we build
    # the `ScalarNonlinearFunction` directly. This is the wire format the
    # parser expects anyway — there is no `crossentropy` JuMP-scalar layer
    # for the user to traverse.
    loss_moi = MOI.ScalarNonlinearFunction(
        :my_crossentropy,
        Any[JuMP.moi_function(Z), target],
    )
    MOI.Nonlinear.set_objective(ad, loss_moi)
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    W_val = [0.3 -0.2; 0.1 0.4]
    x_in = vec(W_val)
    val = MOI.eval_objective(evaluator, x_in)
    @test val ≈ my_crossentropy(my_relu.(W_val * X), target)
    g = zeros(length(x_in))
    MOI.eval_objective_gradient(evaluator, g, x_in)
    ε = 1e-3
    Y_val = W_val * X
    Z_val = my_relu.(Y_val)
    dL_dZ = -target ./ (Z_val .+ ε)
    dZ_dY = Float64.(Y_val .> 0)
    dL_dY = dL_dZ .* dZ_dY
    grad_W = dL_dY * X'
    @test g ≈ vec(grad_W)
    return
end

function test_add_operator_crossentropy_of_relu()
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    model = Model()
    @variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    mode = ArrayDiff.Mode()
    ad = ArrayDiff.model(mode)
    MOI.set(
        ad,
        ArrayDiff.UserDefinedArrayOperator(:my_relu; arity = 1),
        my_relu,
    )
    op_crossentropy = ArrayDiff.add_operator(ad, 2, my_crossentropy)
    @test op_crossentropy isa JuMP.NonlinearOperator
    @test op_crossentropy.head == :my_crossentropy
    Y = W * X
    Z = my_relu.(Y)
    loss = op_crossentropy(Z, target)
    @test loss isa JuMP.NonlinearExpr
    @test loss.head == :my_crossentropy
    @test loss.args[1] === Z
    @test loss.args[2] === target
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(loss))
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    W_val = [0.3 -0.2; 0.1 0.4]
    x_in = vec(W_val)
    val = MOI.eval_objective(evaluator, x_in)
    @test val ≈ my_crossentropy(my_relu.(W_val * X), target)
    g = zeros(length(x_in))
    MOI.eval_objective_gradient(evaluator, g, x_in)
    ε = 1e-3
    Y_val = W_val * X
    Z_val = my_relu.(Y_val)
    dL_dZ = -target ./ (Z_val .+ ε)
    dL_dY = dL_dZ .* Float64.(Y_val .> 0)
    @test g ≈ vec(dL_dY * X')
    return
end

function test_chainrules_broadcasted_relu()
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    model = Model()
    @variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    mode = ArrayDiff.Mode()
    ad = ArrayDiff.model(mode)
    MOI.set(
        ad,
        ArrayDiff.UserDefinedArrayOperator(:my_relu; arity = 1),
        my_relu,
    )
    Y = W * X
    Z = my_relu.(Y)
    @test Z isa ArrayDiff.MatrixExpr
    @test Z.head == :my_relu
    @test Z.broadcasted
    loss = sum(Z)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(loss))
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    W_val = [0.3 -0.2; 0.1 0.4]
    x_in = vec(W_val)
    val = MOI.eval_objective(evaluator, x_in)
    @test val ≈ sum(my_relu.(W_val * X))
    g = zeros(length(x_in))
    MOI.eval_objective_gradient(evaluator, g, x_in)
    # d sum(relu.(W*X)) / dW = (Y .> 0) * X'
    Y_val = W_val * X
    grad_W = Float64.(Y_val .> 0) * X'
    @test g ≈ vec(grad_W)
    return
end

# Generic helper for transformer-style gradient checks: given a JuMP variable
# matrix `x` of shape (seq, d_emb) and a builder that produces a scalar loss,
# verify value+gradient against central finite differences.
function _check_transformer_loss(build_loss; seq = 2, d_emb = 2)
    model = Model()
    @variable(model, x[1:seq, 1:d_emb], container = ArrayDiff.ArrayOfVariables)
    loss = build_loss(x)
    nvar = JuMP.num_variables(model)
    x_pt = randn(nvar)
    _, val, g, evaluator = _eval(model, loss, x_pt; x_grad = x_pt)
    @test isfinite(val)
    @test all(isfinite, g)
    h = 1e-6
    g_fd = zeros(nvar)
    for i in 1:nvar
        xp = copy(x_pt)
        xp[i] += h
        xm = copy(x_pt)
        xm[i] -= h
        g_fd[i] =
            (
                MOI.eval_objective(evaluator, xp) -
                MOI.eval_objective(evaluator, xm)
            ) / (2h)
    end
    @test isapprox(g, g_fd; rtol = 1e-4)
    return
end

# `gelu` mixes `tanh.`, `.+`, `.*`, `.^3`, and scalar-broadcast `Number *
# matrix` / `Number .+ matrix` patterns. Test that the full gelu of a JuMP
# variable matrix is differentiated correctly.
function test_transformer_gelu_gradient()
    return _check_transformer_loss(x -> sum(gelu(x) .^ 2))
end

# Two MLPs in sequence: `m2(m1(x))`. Exercises chain-rule through repeated
# `*` (matrix multiply) and broadcasted `gelu` ops on JuMP variable inputs.
function test_transformer_mlp_chained_gradient()
    d_emb, d_hidden = 2, 3
    c_fc1, c_proj1 = randn(d_emb, d_hidden), randn(d_hidden, d_emb)
    c_fc2, c_proj2 = randn(d_emb, d_hidden), randn(d_hidden, d_emb)
    m1, m2 = MLP(c_fc1, c_proj1), MLP(c_fc2, c_proj2)
    return _check_transformer_loss(x -> sum(m2(m1(x)) .^ 2))
end

# Block-like residual connection `x + mlp(x)`. The transformer Block uses
# this exact `+`-broadcast pattern after each sub-layer; we verify that the
# gradient flows through the sum of identity and a non-linear sub-graph.
function test_transformer_residual_gradient()
    d_emb, d_hidden = 2, 3
    mlp = MLP(randn(d_emb, d_hidden), randn(d_hidden, d_emb))
    return _check_transformer_loss(x -> sum((x .+ mlp(x)) .^ 2))
end

# Stack of two residual MLP blocks, mimicking the structural skeleton of a
# transformer with LayerNorm/Attention swapped out. Exercises a deeper graph
# than the single-block residual test.
function test_transformer_stacked_residual_gradient()
    d_emb, d_hidden = 2, 3
    m1 = MLP(randn(d_emb, d_hidden), randn(d_hidden, d_emb))
    m2 = MLP(randn(d_emb, d_hidden), randn(d_hidden, d_emb))
    function build(x)
        h1 = x .+ m1(x)
        h2 = h1 .+ m2(h1)
        return sum(h2 .^ 2)
    end
    return _check_transformer_loss(build)
end

# Broadcasted `./` against a JuMP matrix variable `W`, with the other
# operand cycling through every shape combination ArrayDiff supports:
# scalar, full matrix, column vector (length rows), row vector (1×cols).
# Loss is `norm(c ./ W)` (variable always in the denominator) and a second
# set of cases with W in the numerator (`W ./ c`); the analytic gradient
# `dexpr_dW .* (c./W) ./ norm(c./W)` is checked against the AD-computed
# gradient elementwise. W is initialized to positive values to avoid the
# division-by-zero blow-up.
function test_broadcast_divide_gradient()
    rows, cols = 2, 3
    c = 2.5
    model = Model()
    @variable(model, W[1:rows, 1:cols], container = ArrayDiff.ArrayOfVariables)
    v = [10.0, 20.0]
    r = [100.0 200.0 300.0]
    M = reshape(collect(11.0:10.0:160.0)[1:(rows*cols)], rows, cols)
    x = Float64.(collect(1:(rows*cols)))
    W_val = reshape(x, rows, cols)
    @testset "$(name)" for (name, expr, ref_mat, dexpr_dW) in [
        # W as denominator: ∂(c ./ W)/∂W_ij = -c_ij / W_ij^2  (c broadcast)
        ("scalar ./ W", c ./ W, c ./ W_val, -c ./ W_val .^ 2),
        ("v ./ W", v ./ W, v ./ W_val, -(v .* ones(rows, cols)) ./ W_val .^ 2),
        ("r ./ W", r ./ W, r ./ W_val, -(ones(rows) .* r) ./ W_val .^ 2),
        ("M ./ W", M ./ W, M ./ W_val, -M ./ W_val .^ 2),
        # W as numerator: ∂(W ./ c)/∂W_ij = 1 / c_ij  (c broadcast)
        ("W ./ scalar", W ./ c, W_val ./ c, fill(1 / c, rows, cols)),
        ("W ./ v", W ./ v, W_val ./ v, 1 ./ (v .* ones(rows, cols))),
        ("W ./ r", W ./ r, W_val ./ r, 1 ./ (ones(rows) .* r)),
        ("W ./ M", W ./ M, W_val ./ M, 1 ./ M),
    ]
        sizes, val, g = _eval(model, LinearAlgebra.norm(expr), x)
        # Tape: norm (k=1, scalar), broadcast `./` (k=2) inheriting (rows, cols)
        # from the result shape, then the two children.
        @test sizes.ndims[1] == 0
        @test sizes.ndims[2] == 2
        b_off = sizes.size_offset[2]
        @test sizes.size[b_off+1] == rows
        @test sizes.size[b_off+2] == cols
        @test val ≈ LinearAlgebra.norm(ref_mat)
        @test g ≈ vec(dexpr_dW .* ref_mat) ./ LinearAlgebra.norm(ref_mat)
    end
    return
end

function _run_infer_sizes_op(f, head::Symbol)
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    model = Model()
    @variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    mode = ArrayDiff.Mode()
    ad = ArrayDiff.model(mode)
    MOI.set(ad, ArrayDiff.UserDefinedArrayOperator(head; arity = 2), f)
    Y = W * X
    Z = my_relu.(Y)  # ensure p > 0 for `my_crossentropy2`'s assertion
    MOI.set(
        ad,
        ArrayDiff.UserDefinedArrayOperator(:my_relu; arity = 1),
        my_relu,
    )
    loss_moi =
        MOI.ScalarNonlinearFunction(head, Any[JuMP.moi_function(Z), target])
    MOI.Nonlinear.set_objective(ad, loss_moi)
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    W_val = [0.3 0.2; 0.1 0.4]
    x_in = vec(W_val)
    val = MOI.eval_objective(evaluator, x_in)
    @test val ≈ f(my_relu.(W_val * X), target)
    g = zeros(length(x_in))
    MOI.eval_objective_gradient(evaluator, g, x_in)
    h = 1e-6
    g_fd = zeros(length(x_in))
    for i in eachindex(x_in)
        xp = copy(x_in)
        xp[i] += h
        xm = copy(x_in)
        xm[i] -= h
        g_fd[i] =
            (
                MOI.eval_objective(evaluator, xp) -
                MOI.eval_objective(evaluator, xm)
            ) / (2h)
    end
    @test isapprox(g, g_fd; rtol = 1e-4)
    return
end

# `my_crossentropy1` has no `infer_sizes` override, so output shape is found
# by probing the function with `zeros(Float64, ...)`. The probe runs fine
# because `log(0 + 1e-3)` is finite.
function test_infer_sizes_default_probe()
    return _run_infer_sizes_op(my_crossentropy1, :my_crossentropy1)
end

# `my_crossentropy2` asserts `all(p .> 0)`, so the default zeros-probe would
# fail. The `infer_sizes` override returns `()` symbolically and avoids
# touching the function — reaching the final `@test` proves the override is
# what ran.
function test_infer_sizes_user_override()
    return _run_infer_sizes_op(my_crossentropy2, :my_crossentropy2)
end

# Whole-array unary op used to exercise the array-output / unary dispatch.
my_double(x::AbstractArray) = 2 .* x

# Covers the unary `(op::NonlinearOperator)(x::AbstractJuMPArray)` dispatch
# and the array-output branch in `_build_user_op_expr` (where `infer_sizes`
# returns a non-empty shape and we build a `GenericArrayExpr` rather than a
# `GenericNonlinearExpr`).
function test_op_unary_jump_array()
    n = 2
    model = Model()
    @variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    op = JuMP.NonlinearOperator(my_double, :my_double)
    result = op(W)
    @test result isa ArrayDiff.MatrixExpr
    @test result.head == :my_double
    @test size(result) == (n, n)
    @test !result.broadcasted
    @test result.args[1] === W
    return
end

# Covers the `(op::NonlinearOperator)(x::Union{Real,AbstractArray{<:Real}}, y::AbstractJuMPArray)`
# dispatch — constant array first, JuMP array second.
function test_op_reversed_args()
    n = 2
    model = Model()
    @variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    target = rand(n, n)
    op = JuMP.NonlinearOperator(my_crossentropy, :my_crossentropy)
    result = op(target, W)
    @test result isa JuMP.NonlinearExpr
    @test result.head == :my_crossentropy
    @test result.args[1] === target
    @test result.args[2] === W
    return
end

function test_matvec_gradient()
    # `W * x` for a 1-D `ArrayOfVariables` `x` exercises the mat-vec branch
    # of `_matmul_reverse!`. Loss is `sum((W*x - target).^2)` so the analytic
    # gradient is `2 * W' * (W*x - target)`.
    m, n = 3, 4
    W = [
        0.4 -0.2 0.1 0.3
        -0.3 0.5 0.2 -0.1
        0.1 0.1 -0.4 0.2
    ]
    target = [0.5, -0.2, 0.1]
    model = Model()
    @variable(model, x[1:n], container = ArrayDiff.ArrayOfVariables)
    y = W * x
    @test y isa ArrayDiff.GenericArrayExpr
    @test ndims(y) == 1
    @test size(y) == (m,)
    @test y.head == :*
    loss = sum((y .- target) .^ 2)
    x_val = [0.6, -0.3, 0.4, -0.1]
    _, val, g, _ = _eval(model, loss, x_val; x_grad = x_val)
    @test val ≈ sum((W * x_val .- target) .^ 2)
    @test g ≈ 2 * W' * (W * x_val .- target)
    return
end

function test_matvec_jump_matrix_times_const_vector_gradient()
    # `W * x` where `W` is an `AbstractJuMPMatrix` and `x` is a constant
    # `Vector`. Loss = sum((W*x - target).^2). Analytic gradient w.r.t. W is
    # `2 * (W*x - target) * x'` (outer product). JuMP stores `W` column-major,
    # so the flat gradient vector is `vec(2 * (W*x - target) * x')`.
    m, n = 3, 4
    x_const = [0.6, -0.3, 0.4, -0.1]
    target = [0.5, -0.2, 0.1]
    model = Model()
    @variable(model, W[1:m, 1:n], container = ArrayDiff.ArrayOfVariables)
    y = W * x_const
    @test y isa ArrayDiff.GenericArrayExpr
    @test ndims(y) == 1
    @test size(y) == (m,)
    @test y.head == :*
    loss = sum((y .- target) .^ 2)
    W_val = [
        0.4 -0.2 0.1 0.3
        -0.3 0.5 0.2 -0.1
        0.1 0.1 -0.4 0.2
    ]
    flat_W = vec(W_val)
    _, val, g, _ = _eval(model, loss, flat_W; x_grad = flat_W)
    @test val ≈ sum((W_val * x_const .- target) .^ 2)
    @test g ≈ vec(2 * (W_val * x_const .- target) * x_const')
    return
end

function test_matvec_jump_matrix_times_jump_vector_gradient()
    # `W * x` where both `W` and `x` are `ArrayOfVariables`. Loss is
    # `sum((W*x - target).^2)`. Gradients: ∂/∂W = 2 (Wx-t) x',
    # ∂/∂x = 2 W' (Wx-t). The flat variable layout is `[vec(W); x]` because
    # `W` is declared first.
    m, n = 3, 4
    target = [0.5, -0.2, 0.1]
    model = Model()
    @variable(model, W[1:m, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, x[1:n], container = ArrayDiff.ArrayOfVariables)
    y = W * x
    @test y isa ArrayDiff.GenericArrayExpr
    @test ndims(y) == 1
    @test size(y) == (m,)
    @test y.head == :*
    loss = sum((y .- target) .^ 2)
    W_val = [
        0.4 -0.2 0.1 0.3
        -0.3 0.5 0.2 -0.1
        0.1 0.1 -0.4 0.2
    ]
    x_val = [0.6, -0.3, 0.4, -0.1]
    flat = [vec(W_val); x_val]
    _, val, g, _ = _eval(model, loss, flat; x_grad = flat)
    @test val ≈ sum((W_val * x_val .- target) .^ 2)
    residual = W_val * x_val .- target
    grad_W = 2 * residual * x_val'
    grad_x = 2 * W_val' * residual
    @test g ≈ [vec(grad_W); grad_x]
    return
end

end  # module

TestJuMP.runtests()
