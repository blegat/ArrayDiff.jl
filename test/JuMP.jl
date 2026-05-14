module TestJuMP

using Test

using JuMP
using ArrayDiff
import LinearAlgebra
import MathOptInterface as MOI

include(joinpath(@__DIR__, "Transformer.jl"))

function runtests()
    for name in names(@__MODULE__; all=true)
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
    sizes = evaluator.backend.objective.expr.sizes
    val = MOI.eval_objective(evaluator, x)
    if VERSION >= v"1.12"
        @test 0 == @allocated MOI.eval_objective(evaluator, x)
    end
    x_grad = T.(collect(1:8))
    g = zero(x)
    MOI.eval_objective_gradient(evaluator, g, x_grad)
    if VERSION >= v"1.12"
        @test 0 == @allocated MOI.eval_objective_gradient(evaluator, g, x_grad)
    end
    MOI.Nonlinear.set_objective(ad, nothing)
    @test isnothing(ad.objective)
    return sizes, val, g
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

function test_broadcast_scalar_matrix_size_inference()
    model = Model()
    @variable(model, W[1:2, 1:3], container = ArrayDiff.ArrayOfVariables)
    mode = ArrayDiff.Mode()
    @testset "$(name)" for (name, expr) in [
        ("scalar .* M", LinearAlgebra.norm(2.5 .* W)),
        ("M .* scalar", LinearAlgebra.norm(W .* 2.5)),
        ("scalar .+ M", LinearAlgebra.norm(2.5 .+ W)),
        ("M .+ scalar", LinearAlgebra.norm(W .+ 2.5)),
        ("scalar .- M", LinearAlgebra.norm(2.5 .- W)),
        ("M .- scalar", LinearAlgebra.norm(W .- 2.5)),
    ]
        ad = ArrayDiff.model(mode)
        MOI.Nonlinear.set_objective(ad, JuMP.moi_function(expr))
        evaluator = MOI.Nonlinear.Evaluator(
            ad,
            mode,
            JuMP.index.(JuMP.all_variables(model)),
        )
        MOI.initialize(evaluator, [:Grad])
        sizes = evaluator.backend.objective.expr.sizes
        # Broadcast node is at index 2; it should inherit the matrix child's
        # (2, 3) shape, not the old `(1, 1)` stub.
        @test sizes.ndims[2] == 2
        broadcast_size_off = sizes.size_offset[2]
        @test sizes.size[broadcast_size_off+1] == 2
        @test sizes.size[broadcast_size_off+2] == 3
        # And the scalar leaf among the children stays ndims=0.
        @test 0 in sizes.ndims[3:4]
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
            (MOI.eval_objective(evaluator, xp) -
             MOI.eval_objective(evaluator, xm)) / (2h)
    end
    @test isapprox(g, g_fd; rtol = 1e-4)
    return
end

end  # module

TestJuMP.runtests()
