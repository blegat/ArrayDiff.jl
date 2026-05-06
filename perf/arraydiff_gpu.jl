# Benchmark `ArrayDiff` GPU gradient computation against the hand-written
# CUDA `reverse_diff` reference (the same one as `perf/cuda_vs_pytorch.jl`,
# but in `Float64` so it matches `ArrayDiff`'s tape eltype).
#
# What this measures
# ------------------
# A 2-layer MLP `loss = sum((W2 * tanh.(W1 * X) - y).^2) / n`, where `W1` is
# the only `@variable` (so `MOI.eval_objective_gradient` returns ∂loss/∂W1)
# and `W2`, `X`, `y` are constants.
#
#   * Hand-CUDA Float64: `reverse_diff(W1, W2, X, y)` — the same hand-written
#     formula as the existing perf script, just promoted to `Float64` so the
#     numbers are directly comparable to ArrayDiff.
#
#   * ArrayDiff CPU: same model with `Mode{Vector{Float64}}()` — sanity check
#     for what the existing CPU AD path costs at this problem size.
#
#   * ArrayDiff GPU: same model with `Mode{CuVector{Float64}}()` — the path
#     this benchmark exists to measure.
#
# Caveats (read this before reading the numbers)
# ----------------------------------------------
# The current `ArrayDiff` GPU path still does scalar reads/writes on the
# tape for every `NODE_VARIABLE` (one per `W1[i,j]`) and every `NODE_VALUE`
# (one per scalar inside the `vcat(row(...), row(...))` expansion of
# constant matrices `W2`, `X`, `y`). On a `CuArray` each scalar load/store
# is a separate `cudaMemcpy`, so for `h = 4096` and `d = 13` that's tens of
# thousands of round-trips per call. We expect the GPU path to be much
# slower than the hand-written CUDA reference until those leaves are
# loaded/extracted in bulk — see the `NODE_VARIABLE_BLOCK` /
# `NODE_VALUE_BLOCK` follow-up.
#
# The whole `optimize!` loop is wrapped in `CUDA.@allowscalar` so the
# scalar paths don't error out under GPUArrays' default scalar policy.
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
# Hand-written CUDA Float64 reverse_diff — same formulas as
# `perf/cuda_vs_pytorch.jl::forward_pass`/`reverse_diff`, but in `Float64`
# and *without* the `/ n` scaling. ArrayDiff's `:/` operator currently
# has a latent bug with non-scalar tape offsets (it reads
# `forward_storage[node_idx]` instead of `forward_storage[storage_offset[node_idx]+1]`),
# so we drop the constant scaling factor on both sides — it doesn't change
# what the gradient pathway exercises.
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
# ArrayDiff path. Builds a JuMP model with `W1` as the only `@variable` and
# `W2 / X / y` baked in as constant matrices, parses to ArrayDiff, and
# returns an evaluator + a closure that computes ∂loss/∂W1 in `g`.
# -------------------------------------------------------------------------
function build_arraydiff(
    W2::Matrix{Float64},
    X::Matrix{Float64},
    y::Matrix{Float64},
    h::Int,
    d::Int,
    n::Int,
    mode::ArrayDiff.Mode,
)
    model = JuMP.Model()
    @variable(model, W1[1:h, 1:d], container = ArrayDiff.ArrayOfVariables)
    Y = W2 * tanh.(W1 * X)
    diff = Y .- y
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

# CPU path — `Mode()` defaults to `Vector{Float64}` storage, no `@allowscalar`
# needed.
function arraydiff_grad_cpu!(g, evaluator, x)
    MOI.eval_objective_gradient(evaluator, g, x)
    return g
end

# GPU path — `Mode{CuVector{Float64}}()`. The current implementation still
# uses scalar tape access for `NODE_VARIABLE`/`NODE_VALUE` leaves, so the
# call needs `@allowscalar` to get past GPUArrays' scalar-indexing policy.
function arraydiff_grad_gpu!(g, evaluator, x)
    CUDA.@allowscalar MOI.eval_objective_gradient(evaluator, g, x)
    return g
end

# -------------------------------------------------------------------------
# One (h, d, n) sweep.
# -------------------------------------------------------------------------
function run_one(; h::Int, d::Int = 13, n::Int = 178, rtol::Float64 = 1e-6)
    println("\n" * "="^72)
    @printf "h = %d, d = %d, n = %d\n" h d n
    println("="^72)

    Random.seed!(0)
    # `W2` and `y` are deliberately given 2 output rows (instead of the
    # 1-row used in `perf/cuda_vs_pytorch.jl`). The current `:vcat`
    # forward unpacks `idx1, idx2 = children_indices` and so requires at
    # least two rows; a single-row matrix would be parsed as a `vcat`
    # with one child and would error. Using 2 rows changes nothing about
    # the gradient pathway being measured.
    W1 = randn(Float64, h, d)
    W2 = randn(Float64, 2, h)
    X = randn(Float64, d, n)
    y = randn(Float64, 2, n)

    # Reference: hand-written CUDA in Float64.
    W1g = CuArray(W1)
    W2g = CuArray(W2)
    Xg = CuArray(X)
    yg = CuArray(y)
    grad_ref = Array(reverse_diff(W1g, W2g, Xg, yg))
    CUDA.synchronize()

    # ArrayDiff CPU.
    print("ArrayDiff CPU build (h=$h) ... ");
    flush(stdout)
    t_cpu_build =
        @elapsed ev_cpu = build_arraydiff(W2, X, y, h, d, n, ArrayDiff.Mode())
    @printf "%.2f s\n" t_cpu_build
    x_cpu = vec(W1)
    g_cpu = zeros(Float64, length(x_cpu))
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
        ArrayDiff.Mode{CUDA.CuVector{Float64}}(),
    )
    @printf "%.2f s\n" t_gpu_build
    x_gpu = CUDA.CuVector{Float64}(vec(W1))
    g_gpu = CUDA.zeros(Float64, length(x_gpu))
    arraydiff_grad_gpu!(g_gpu, ev_gpu, x_gpu)
    CUDA.synchronize()
    grad_gpu = reshape(Array(g_gpu), h, d)

    # Numerical equivalence.
    for (name, g) in [("ArrayDiff CPU", grad_cpu), ("ArrayDiff GPU", grad_gpu)]
        maxdiff = maximum(abs.(grad_ref .- g))
        relmag = maxdiff / max(maximum(abs.(grad_ref)), eps(Float64))
        ok = isapprox(grad_ref, g; rtol = rtol)
        @printf "%-15s vs hand-CUDA: max|Δ| = %.3e (rel %.2e)  match=%s\n" name maxdiff relmag ok
    end

    println("\n--- benchmark (median of N samples, post-sync) ---")
    bj = @benchmark begin
        reverse_diff($W1g, $W2g, $Xg, $yg)
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
    @printf "Hand-CUDA Float64 : median %12.2f µs\n" 1e-3 * median(bj).time
    @printf "ArrayDiff CPU     : median %12.2f µs\n" 1e-3 * median(bcpu).time
    @printf "ArrayDiff GPU     : median %12.2f µs\n" 1e-3 * median(bgpu).time

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
