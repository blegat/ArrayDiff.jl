# Compare hardcoded CUDA.jl forward+reverse for a 2-layer MLP gradient
# against PyTorch (via PythonCall) doing the equivalent loss + .backward().
#
# Goals
#   1. Numerical equality of ∂L/∂W1 between the two paths.
#   2. Per-kernel CUDA trace from each, side by side, so we can see whether
#      they issue the same GPU operations under the hood.
#   3. Wall-clock benchmark per hidden size h.
#
# Run
#   cd ~/.julia/dev/ArrayDiff
#   julia --project=perf -e 'using Pkg; Pkg.instantiate()'
#   julia --project=perf perf/cuda_vs_pytorch.jl
#
# Formulas are transcribed verbatim from autodiff.jl:398-426.
# Loss: L = sum((W2 * tanh(W1 * X) - y) .^ 2) / size(y, 2)
# Returned grad: ∂L/∂W1 (∈ R^{h×d}).

using Random, LinearAlgebra, Printf
using CUDA
using BenchmarkTools
using PythonCall

# -------------------------------------------------------------------------
# Hardcoded CUDA.jl path
# -------------------------------------------------------------------------
function forward_pass(W1, W2, X, y)
    y_1 = tanh.(W1 * X)
    J_1 = 1 .- y_1 .^ 2
    J_2 = 2 .* (W2 * y_1 .- y) ./ size(y, 2)
    return y_1, J_1, J_2
end

function reverse_diff(W1, W2, X, y)
    _, J_1, J_2 = forward_pass(W1, W2, X, y)
    return (J_1 .* (W2' * J_2)) * X'
end

# -------------------------------------------------------------------------
# PyTorch path
# -------------------------------------------------------------------------
const torch    = pyimport("torch")
const np       = pyimport("numpy")
const profiler = pyimport("torch.profiler")

# Build torch tensors once and reuse them across benchmark iterations,
# mirroring how the Julia path passes already-on-GPU CuArrays.
function build_torch_tensors(W1::Matrix, W2::Matrix, X::Matrix, y::Matrix)
    npW1 = np.ascontiguousarray(np.asarray(PyArray(W1)))
    npW2 = np.ascontiguousarray(np.asarray(PyArray(W2)))
    npX  = np.ascontiguousarray(np.asarray(PyArray(X)))
    npY  = np.ascontiguousarray(np.asarray(PyArray(y)))
    W1t = torch.from_numpy(npW1).to("cuda").requires_grad_(true)
    W2t = torch.from_numpy(npW2).to("cuda")
    Xt  = torch.from_numpy(npX).to("cuda")
    yt  = torch.from_numpy(npY).to("cuda")
    return W1t, W2t, Xt, yt
end

function pytorch_grad(W1t, W2t, Xt, yt)
    y1   = torch.tanh(torch.matmul(W1t, Xt))
    diff = torch.matmul(W2t, y1) - yt
    n    = pyconvert(Int, yt.shape[1])
    loss = (diff * diff).sum() / n
    grad = torch.autograd.grad(loss, W1t)[0]
    return grad
end

torch_to_julia(t) = pyconvert(Array, t.detach().cpu().numpy())

# -------------------------------------------------------------------------
# Trace helpers
# -------------------------------------------------------------------------
function julia_trace(f)
    # Warmup so JIT + cuBLAS handle init don't show up.
    f(); CUDA.synchronize()
    return CUDA.@profile trace = true begin
        f()
        CUDA.synchronize()
    end
end

function pytorch_trace(f)
    f(); torch.cuda.synchronize()  # warmup
    ProfilerActivity = profiler.ProfilerActivity
    prof = profiler.profile(activities = pylist([ProfilerActivity.CUDA]))
    prof.__enter__()
    try
        f()
        torch.cuda.synchronize()
    finally
        prof.__exit__(pybuiltins.None, pybuiltins.None, pybuiltins.None)
    end
    return prof.key_averages().table(sort_by = "cuda_time_total")
end

# -------------------------------------------------------------------------
# Benchmark + verify for one (h, d, n)
# -------------------------------------------------------------------------
function run_one(; h::Int, d::Int = 13, n::Int = 178, rtol::Float32 = 1f-3)
    println("\n" * "="^72)
    @printf "h = %d, d = %d, n = %d\n" h d n
    println("="^72)

    Random.seed!(0)
    W1 = randn(Float32, h, d)
    W2 = randn(Float32, 1, h)
    X  = randn(Float32, d, n)
    y  = randn(Float32, 1, n)

    # ----- Julia / CUDA.jl -----
    W1g = CuArray(W1); W2g = CuArray(W2); Xg = CuArray(X); yg = CuArray(y)
    grad_julia = Array(reverse_diff(W1g, W2g, Xg, yg))
    CUDA.synchronize()

    # ----- PyTorch -----
    W1t, W2t, Xt, yt = build_torch_tensors(W1, W2, X, y)
    grad_pytorch = torch_to_julia(pytorch_grad(W1t, W2t, Xt, yt))
    torch.cuda.synchronize()

    # ----- Numerical equivalence -----
    maxdiff = maximum(abs.(grad_julia .- grad_pytorch))
    relmag  = maxdiff / max(maximum(abs.(grad_julia)), eps(Float32))
    @printf "max|Δ|         = %.3e\n" maxdiff
    @printf "max|Δ| / scale = %.3e\n" relmag
    ok = isapprox(grad_julia, grad_pytorch; rtol = rtol, atol = 1f-4)
    println("gradients match: ", ok)

    # ----- Benchmarks -----
    println("\n--- benchmark (median over many samples, includes CUDA.synchronize) ---")
    bj = @benchmark begin
        reverse_diff($W1g, $W2g, $Xg, $yg)
        CUDA.synchronize()
    end
    bp = @benchmark begin
        pytorch_grad($W1t, $W2t, $Xt, $yt)
        $torch.cuda.synchronize()
    end
    @printf "Julia (CUDA.jl) : median %8.3f µs\n" 1e-3 * median(bj).time
    @printf "PyTorch eager   : median %8.3f µs\n" 1e-3 * median(bp).time

    # ----- CUDA traces -----
    println("\n--- CUDA trace: Julia / CUDA.jl ---")
    show(stdout, "text/plain", julia_trace(() -> reverse_diff(W1g, W2g, Xg, yg)))
    println()

    println("\n--- CUDA trace: PyTorch ---")
    println(pytorch_trace(() -> pytorch_grad(W1t, W2t, Xt, yt)))

    return nothing
end

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
function main()
    if !CUDA.functional()
        error("CUDA is not functional in this environment.")
    end
    if !pyconvert(Bool, torch.cuda.is_available())
        error("PyTorch reports CUDA is not available.")
    end
    println("CUDA.jl device : ", CUDA.name(CUDA.device()))
    println("PyTorch device : ", pyconvert(String, torch.cuda.get_device_name(0)))

    for h in (16, 256, 4096)
        run_one(; h = h)
    end
end

main()
