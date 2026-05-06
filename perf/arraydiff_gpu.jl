# Benchmark `ArrayDiff` GPU gradient computation against the hand-written
# CUDA `reverse_diff` reference (the same one as `perf/cuda_vs_pytorch.jl`).
#
# Float32 throughout — this is the precision that maps onto the tensor cores
# / SP pipeline a GPU is actually optimized for, and matches what a typical
# ML workload (PyTorch default) feeds the device.
#
# What this measures
# ------------------
# A 2-layer MLP `loss = sum((W2 * tanh.(W1 * X) - y).^2) / n`, where `W1` is
# the only `@variable` (so `MOI.eval_objective_gradient` returns ∂loss/∂W1)
# and `W2`, `X`, `y` are constants.
#
#   * Hand-CUDA Float32 (allocating): `reverse_diff(W1, W2, X, y)` — same
#     hand-written formula as `perf/cuda_vs_pytorch.jl`, allocates fresh
#     intermediates per call. This is the "naive" hand-tuned baseline.
#
#   * Hand-CUDA Float32 (preallocated): `reverse_diff_prealloc!(buf, ...)` —
#     same arithmetic but every intermediate lives in a `HandPrealloc` buffer
#     that is reused across calls. This is the apples-to-apples comparison
#     against ArrayDiff, which preallocates its tape once at
#     `MOI.initialize` and never allocates again.
#
#   * ArrayDiff CPU: same model with `Mode{Vector{Float32}}()` — sanity check
#     for what the existing CPU AD path costs at this problem size.
#
#   * ArrayDiff GPU: same model with `Mode{CuVector{Float32}}()` — the path
#     this benchmark exists to measure.
#
# Run
# ---
#   cd ~/.julia/dev/ArrayDiff
#   julia --project=perf -e 'using Pkg; Pkg.instantiate()'
#   julia --project=perf perf/arraydiff_gpu.jl

using Random, LinearAlgebra, Printf
using BenchmarkTools
using CUDA
using JuMP
using ArrayDiff
import MathOptInterface as MOI

# -------------------------------------------------------------------------
# Hand-written CUDA reverse_diff — same formulas as
# `perf/cuda_vs_pytorch.jl::forward_pass`/`reverse_diff`, but *without* the
# `/ n` scaling. ArrayDiff's `:/` operator currently has a latent bug with
# non-scalar tape offsets (it reads `forward_storage[node_idx]` instead of
# `forward_storage[storage_offset[node_idx]+1]`), so we drop the constant
# scaling factor on both sides — it doesn't change what the gradient
# pathway exercises.
#
# These functions are eltype-generic (they're just dotted broadcasts and
# `*`), so the same code runs at `Float32` and `Float64` depending on the
# eltype of the `CuArray` inputs.
# -------------------------------------------------------------------------
function forward_pass(W1, W2, X, y)
    y_1 = tanh.(W1 * X)
    J_1 = 1 .- y_1 .^ 2
    J_2 = 2 .* (W2 * y_1 .- y)
    return y_1, J_1, J_2
end

function reverse_diff(W1, W2, X, y)
    _, J_1, J_2 = forward_pass(W1, W2, X, y)
    return (J_1 .* (W2' * J_2)) * X'
end

# -------------------------------------------------------------------------
# Hand-written CUDA reverse_diff — preallocated variant.
#
# `reverse_diff` above allocates ~6 fresh `CuArray`s per call (one for
# `tanh.(...)`, one for `1 .- y_1.^2`, one for `2 .* (...)`, plus three for
# the chained `*` / `.*` / `*` in the reverse step). At h = 4096 each is a
# multi-MB matrix, and the allocator overhead dominates the actual GEMM
# compute. To be a fair "hand-tuned" baseline against ArrayDiff (which
# allocates its tape exactly once at `MOI.initialize` and reuses it), we
# pre-allocate all intermediates in `HandPrealloc` and run forward + reverse
# with `mul!` + in-place broadcasts.
#
# Buffer reuse: `y_1` doubles as the `W1 * X` output, then is overwritten
# in place by `tanh`. `J_2` doubles as the `W2 * y_1` output and is then
# overwritten by `2 .* (J_2 - y)`. `W2T_J2` is scaled in place by `J_1`.
# Five device buffers cover everything.
# -------------------------------------------------------------------------
struct HandPrealloc{T<:Real}
    y_1::CuArray{T,2}      # h × n   (= tanh.(W1 * X))
    J_1::CuArray{T,2}      # h × n   (= 1 .- y_1.^2)
    J_2::CuArray{T,2}      # out × n (= 2 .* (W2*y_1 - y))
    W2T_J2::CuArray{T,2}   # h × n   (= W2' * J_2, then ⊙= J_1)
    grad::CuArray{T,2}     # h × d   (= W2T_J2 * X')
end

function HandPrealloc{T}(h::Int, d::Int, n::Int, out_dim::Int) where {T}
    return HandPrealloc{T}(
        CuArray{T}(undef, h, n),
        CuArray{T}(undef, h, n),
        CuArray{T}(undef, out_dim, n),
        CuArray{T}(undef, h, n),
        CuArray{T}(undef, h, d),
    )
end

function reverse_diff_prealloc!(buf::HandPrealloc, W1, W2, X, y)
    # Forward
    LinearAlgebra.mul!(buf.y_1, W1, X)            # y_1 = W1 * X
    buf.y_1 .= tanh.(buf.y_1)                     # y_1 = tanh.(y_1)
    buf.J_1 .= 1 .- buf.y_1 .^ 2                  # J_1 = 1 .- y_1.^2
    LinearAlgebra.mul!(buf.J_2, W2, buf.y_1)      # J_2 = W2 * y_1
    buf.J_2 .= 2 .* (buf.J_2 .- y)                # J_2 = 2 .* (W2*y_1 - y)
    # Reverse — (J_1 .* (W2' * J_2)) * X'
    LinearAlgebra.mul!(buf.W2T_J2, W2', buf.J_2)  # W2T_J2 = W2' * J_2
    buf.W2T_J2 .= buf.J_1 .* buf.W2T_J2           # W2T_J2 ⊙= J_1
    LinearAlgebra.mul!(buf.grad, buf.W2T_J2, X')  # grad = W2T_J2 * X'
    return buf.grad
end

# -------------------------------------------------------------------------
# ArrayDiff path. Builds a JuMP model with `W1` as the only `@variable` and
# `W2 / X / y` baked in as constant matrices, parses to ArrayDiff, and
# returns an evaluator + a closure that computes ∂loss/∂W1 in `g`.
# -------------------------------------------------------------------------
function build_arraydiff(
    W2::Matrix{T},
    X::Matrix{T},
    y::Matrix{T},
    h::Int,
    d::Int,
    n::Int,
    mode::ArrayDiff.Mode,
) where {T<:Real}
    # JuMP works in Float64 internally; ArrayDiff converts the parsed
    # `Expression{Float64}` constants into the `Mode`'s eltype (here Float32)
    # when filling its tape. We promote constants to Float64 for the JuMP
    # build to keep that conversion explicit / one-way.
    W2_64 = Float64.(W2)
    X_64 = Float64.(X)
    y_64 = Float64.(y)
    model = JuMP.Model()
    @variable(model, W1[1:h, 1:d], container = ArrayDiff.ArrayOfVariables)
    Y = W2_64 * tanh.(W1 * X_64)
    diff = Y .- y_64
    loss = sum(diff .^ 2)  # `/ n` dropped — see `forward_pass` comment.
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(loss))
    evaluator = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(evaluator, [:Grad])
    return evaluator
end

function arraydiff_grad_cpu!(g, evaluator, x)
    MOI.eval_objective_gradient(evaluator, g, x)
    return g
end

# `@allowscalar` covers the residual scalar leaves that the BLOCK rewrite
# can't fold (e.g. a stand-alone `NODE_VALUE` inside a non-block subterm).
# The hot path is bulk; this is a guard so we don't trip GPUArrays' policy
# on the leftovers.
function arraydiff_grad_gpu!(g, evaluator, x)
    CUDA.@allowscalar MOI.eval_objective_gradient(evaluator, g, x)
    return g
end

# -------------------------------------------------------------------------
# One (h, d, n) sweep.
# -------------------------------------------------------------------------
function run_one(; h::Int, d::Int = 13, n::Int = 178, rtol::Float32 = 1.0f-3)
    println("\n" * "="^72)
    @printf "h = %d, d = %d, n = %d  (Float32)\n" h d n
    println("="^72)

    T = Float32
    Random.seed!(0)
    # `W2` and `y` are deliberately given 2 output rows (instead of the
    # 1-row used in `perf/cuda_vs_pytorch.jl`). The current `:vcat`
    # forward unpacks `idx1, idx2 = children_indices` and so requires at
    # least two rows; a single-row matrix would be parsed as a `vcat`
    # with one child and would error. Using 2 rows changes nothing about
    # the gradient pathway being measured.
    out_dim = 2
    W1 = randn(T, h, d)
    W2 = randn(T, out_dim, h)
    X = randn(T, d, n)
    y = randn(T, out_dim, n)

    # Reference: hand-written CUDA (allocating).
    W1g = CuArray(W1)
    W2g = CuArray(W2)
    Xg = CuArray(X)
    yg = CuArray(y)
    grad_ref = Array(reverse_diff(W1g, W2g, Xg, yg))
    CUDA.synchronize()

    # Hand-CUDA preallocated. Same arithmetic, but every intermediate is
    # owned by `hand_buf` and reused across calls.
    hand_buf = HandPrealloc{T}(h, d, n, out_dim)
    grad_prealloc = Array(reverse_diff_prealloc!(hand_buf, W1g, W2g, Xg, yg))
    CUDA.synchronize()

    # ArrayDiff CPU.
    print("ArrayDiff CPU build (h=$h) ... ");
    flush(stdout)
    t_cpu_build = @elapsed ev_cpu =
        build_arraydiff(W2, X, y, h, d, n, ArrayDiff.Mode{Vector{T}}())
    @printf "%.2f s\n" t_cpu_build
    x_cpu = vec(W1)
    g_cpu = zeros(T, length(x_cpu))
    arraydiff_grad_cpu!(g_cpu, ev_cpu, x_cpu)
    grad_cpu = reshape(g_cpu, h, d)

    # ArrayDiff GPU. Both `x` and `g` live on the GPU — same convention a
    # GPU-resident solver (e.g. one whose ADAM step is on `CuVector`) would
    # use: the AD tape, the input vector, and the gradient buffer all stay
    # on the device, so there's no D2H round-trip on the gradient hot path.
    print("ArrayDiff GPU build (h=$h) ... ");
    flush(stdout)
    t_gpu_build = @elapsed ev_gpu = build_arraydiff(
        W2,
        X,
        y,
        h,
        d,
        n,
        ArrayDiff.Mode{CUDA.CuVector{T}}(),
    )
    @printf "%.2f s\n" t_gpu_build
    x_gpu = CUDA.CuVector{T}(vec(W1))
    g_gpu = CUDA.zeros(T, length(x_gpu))
    arraydiff_grad_gpu!(g_gpu, ev_gpu, x_gpu)
    CUDA.synchronize()
    grad_gpu = reshape(Array(g_gpu), h, d)

    # Numerical equivalence (rtol loosened for Float32 reductions).
    candidates = [
        ("Hand-CUDA prealloc", grad_prealloc),
        ("ArrayDiff CPU", grad_cpu),
        ("ArrayDiff GPU", grad_gpu),
    ]
    for (name, g) in candidates
        maxdiff = maximum(abs.(grad_ref .- g))
        relmag = maxdiff / max(maximum(abs.(grad_ref)), eps(T))
        ok = isapprox(grad_ref, g; rtol = rtol)
        @printf "%-20s vs hand-CUDA alloc: max|Δ| = %.3e (rel %.2e)  match=%s\n" name maxdiff relmag ok
    end

    println("\n--- benchmark (median of N samples, post-sync) ---")
    bj = @benchmark begin
        reverse_diff($W1g, $W2g, $Xg, $yg)
        CUDA.synchronize()
    end samples = 30 evals = 1 seconds = 10
    bjp = @benchmark begin
        reverse_diff_prealloc!($hand_buf, $W1g, $W2g, $Xg, $yg)
        CUDA.synchronize()
    end samples = 30 evals = 1 seconds = 10
    bcpu = @benchmark begin
        arraydiff_grad_cpu!($g_cpu, $ev_cpu, $x_cpu)
    end samples = 30 evals = 1 seconds = 10
    # `seconds = 60` for the GPU path because, at h = 4096, the
    # scalar-cudaMemcpy path can run into the 10s cap on a single sample at
    # warmup; bump the budget so we get a reliable median.
    bgpu = @benchmark begin
        arraydiff_grad_gpu!($g_gpu, $ev_gpu, $x_gpu)
        CUDA.synchronize()
    end samples = 30 evals = 1 seconds = 60
    @printf "Hand-CUDA alloc    : median %12.2f µs\n" 1e-3 * median(bj).time
    @printf "Hand-CUDA prealloc : median %12.2f µs\n" 1e-3 * median(bjp).time
    @printf "ArrayDiff CPU      : median %12.2f µs\n" 1e-3 * median(bcpu).time
    @printf "ArrayDiff GPU      : median %12.2f µs\n" 1e-3 * median(bgpu).time

    return nothing
end

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
function main()
    if !CUDA.functional()
        error("CUDA is not functional in this environment.")
    end
    CUDA.math_mode!(CUDA.FAST_MATH)
    println(
        "CUDA.jl device : ",
        CUDA.name(CUDA.device()),
        "  (math_mode=FAST_MATH)",
    )
    for h in (16, 256, 4096)
        run_one(; h = h)
        GC.gc(true)
        CUDA.reclaim()
    end
end

main()
