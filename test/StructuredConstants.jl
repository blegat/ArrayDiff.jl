module TestStructuredConstants

# Tests for `NODE_ARRAY_VALUE`: constant `AbstractArray`s that are not dense
# `Array`s are kept by reference in `Expression.arrays` instead of being
# serialized on the AD tape, and matmul nodes call `LinearAlgebra.mul!`
# directly on them so their specialized (sparse / structured) methods apply.

using Test

using JuMP
using ArrayDiff
import LinearAlgebra
import MathOptInterface as MOI
import SparseArrays

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

# A minimal custom matrix type: `y = x[idx]` as a linear operator. Each row
# has exactly one entry equal to 1, in column `idx[row]`. It only implements
# the two `mul!` methods ArrayDiff needs, so hitting any other code path
# (for example, an attempt to serialize it densely) would error.
struct SelectionMatrix{V<:AbstractVector{<:Integer}} <: AbstractMatrix{Float64}
    idx::V
    ncol::Int
end

Base.size(A::SelectionMatrix) = (length(A.idx), A.ncol)

# Dense copy for the reference computations of the tests. We deliberately
# don't implement `getindex`: evaluation must never index the matrix, so
# leaving it out proves that only the two `mul!` methods are used.
function _matrix(A::SelectionMatrix)
    B = zeros(size(A))
    for (i, j) in enumerate(A.idx)
        B[i, j] = 1.0
    end
    return B
end

function LinearAlgebra.mul!(
    y::AbstractVector,
    A::SelectionMatrix,
    x::AbstractVector,
)
    y .= view(x, A.idx)
    return y
end

function LinearAlgebra.mul!(
    y::AbstractVector,
    At::LinearAlgebra.Transpose{Float64,<:SelectionMatrix},
    w::AbstractVector,
)
    A = parent(At)
    fill!(y, 0.0)
    for (i, j) in enumerate(A.idx)
        y[j] += w[i]
    end
    return y
end

function _gradient(model, obj, x)
    mode = ArrayDiff.Mode{Vector{Float64}}()
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(obj))
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    val = MOI.eval_objective(evaluator, x)
    g = zero(x)
    MOI.eval_objective_gradient(evaluator, g, x)
    return val, g
end

function test_parse_sparse_constant_by_reference()
    n = 4
    A = SparseArrays.sprand(n, n, 0.5) + LinearAlgebra.I
    model = JuMP.Model()
    @variable(model, x[1:n], container = ArrayDiff.ArrayOfVariables)
    expr = A * x
    f = JuMP.moi_function(sum(expr .^ 2))
    mode = ArrayDiff.Mode{Vector{Float64}}()
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, f)
    # The sparse matrix must be stored by reference, not copied to `values`.
    obj = ad.objective
    @test length(obj.arrays) == 1
    @test obj.arrays[1] === A
    @test any(node -> node.type == ArrayDiff.NODE_ARRAY_VALUE, obj.nodes)
    return
end

function test_sparse_matvec_lhs_gradient()
    n = 5
    A = SparseArrays.sprand(n, n, 0.4) + LinearAlgebra.I
    A_dense = Matrix(A)
    x_val = collect(1.0:n)
    model = JuMP.Model()
    @variable(model, x[1:n], container = ArrayDiff.ArrayOfVariables)
    obj_sparse = sum((A * x) .^ 2)
    obj_dense = sum((A_dense * x) .^ 2)
    val_s, g_s = _gradient(model, obj_sparse, x_val)
    val_d, g_d = _gradient(model, obj_dense, x_val)
    @test val_s ≈ val_d
    @test g_s ≈ g_d
    # Reference: ∇ sum((Ax).^2) = 2 A' A x
    @test g_s ≈ 2 * A_dense' * (A_dense * x_val)
    return
end

function test_custom_selection_matrix_gradient()
    n = 4
    idx = [2, 4, 1, 1, 3]
    A = SelectionMatrix(idx, n)
    A_dense = _matrix(A)
    x_val = [0.5, -1.0, 2.0, 3.0]
    c = [1.0, 2.0, 3.0, 4.0, 5.0]
    model = JuMP.Model()
    @variable(model, x[1:n], container = ArrayDiff.ArrayOfVariables)
    obj_custom = sum(c .* (A * x) .^ 2)
    obj_dense = sum(c .* (A_dense * x) .^ 2)
    val_c, g_c = _gradient(model, obj_custom, x_val)
    val_d, g_d = _gradient(model, obj_dense, x_val)
    @test val_c ≈ val_d
    @test g_c ≈ g_d
    @test g_c ≈ 2 * A_dense' * (c .* (A_dense * x_val))
    return
end

function test_sparse_matmat_rhs_gradient()
    m, n = 3, 4
    B = SparseArrays.sprand(n, n, 0.5) + LinearAlgebra.I
    B_dense = Matrix(B)
    W_val = reshape(collect(1.0:(m*n)), m, n)
    model = JuMP.Model()
    @variable(model, W[1:m, 1:n], container = ArrayDiff.ArrayOfVariables)
    obj_sparse = sum((W * B) .^ 2)
    obj_dense = sum((W * B_dense) .^ 2)
    x_val = vec(W_val)
    val_s, g_s = _gradient(model, obj_sparse, x_val)
    val_d, g_d = _gradient(model, obj_dense, x_val)
    @test val_s ≈ val_d
    @test g_s ≈ g_d
    # Reference: ∇_W sum((WB).^2) = 2 (WB) B'
    @test reshape(g_s, m, n) ≈ 2 * (W_val * B_dense) * B_dense'
    return
end

function test_sparse_residual_jtprod()
    n, m = 6, 4
    A = SparseArrays.sprand(m, n, 0.5)
    A_dense = Matrix(A)
    b = collect(range(-1.0, 1.0; length = m))
    f_sparse = x -> A * x .+ b
    f_dense = x -> A_dense * x .+ b
    x_val = sin.(1:n)
    v = cos.(1:m)
    ev_s = ArrayDiff.evaluator(f_sparse, n)
    ev_d = ArrayDiff.evaluator(f_dense, n)
    F_s, F_d = zeros(m), zeros(m)
    ArrayDiff.eval_residual!(ev_s, F_s, x_val)
    ArrayDiff.eval_residual!(ev_d, F_d, x_val)
    @test F_s ≈ F_d
    @test F_s ≈ A_dense * x_val .+ b
    Jtv_s, Jtv_d = zeros(n), zeros(n)
    ArrayDiff.eval_residual_jtprod!(ev_s, Jtv_s, x_val, v)
    ArrayDiff.eval_residual_jtprod!(ev_d, Jtv_d, x_val, v)
    @test Jtv_s ≈ Jtv_d
    @test Jtv_s ≈ A_dense' * v
    return
end

function test_sparse_with_nonlinear_chain()
    # Mimics the AC-OPF polar structure: gather, sin/cos of differences,
    # elementwise products, scatter back.
    nbus, nbr = 4, 5
    from = [1, 1, 2, 3, 4]
    to = [2, 3, 3, 4, 1]
    F = SelectionMatrix(from, nbus)
    T_ = SelectionMatrix(to, nbus)
    F_dense, T_dense = _matrix(F), _matrix(T_)
    c = [0.3, -0.5, 1.1, 0.7, -0.2]
    build = (F1, T1) -> function (va)
        d = F1 * va .- T1 * va
        return c .* sin.(d) .+ cos.(d)
    end
    va_val = [0.0, 0.1, -0.2, 0.3]
    v = collect(1.0:nbr)
    ev_c = ArrayDiff.evaluator(build(F, T_), nbus)
    ev_d = ArrayDiff.evaluator(build(F_dense, T_dense), nbus)
    F_c, F_d = zeros(nbr), zeros(nbr)
    ArrayDiff.eval_residual!(ev_c, F_c, va_val)
    ArrayDiff.eval_residual!(ev_d, F_d, va_val)
    d_val = va_val[from] .- va_val[to]
    @test F_c ≈ F_d
    @test F_c ≈ c .* sin.(d_val) .+ cos.(d_val)
    Jtv_c, Jtv_d = zeros(nbus), zeros(nbus)
    ArrayDiff.eval_residual_jtprod!(ev_c, Jtv_c, va_val, v)
    ArrayDiff.eval_residual_jtprod!(ev_d, Jtv_d, va_val, v)
    @test Jtv_c ≈ Jtv_d
    # Finite-difference check of J'v.
    h = 1e-7
    Jtv_fd = map(eachindex(va_val)) do i
        va_p = copy(va_val)
        va_p[i] += h
        va_m = copy(va_val)
        va_m[i] -= h
        fp = va -> c .* sin.(va[from] .- va[to]) .+ cos.(va[from] .- va[to])
        return LinearAlgebra.dot(v, (fp(va_p) .- fp(va_m)) ./ (2h))
    end
    @test Jtv_c ≈ Jtv_fd atol = 1e-6
    return
end

end

TestStructuredConstants.runtests()
