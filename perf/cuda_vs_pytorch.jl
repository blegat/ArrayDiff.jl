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

# Define eager + torch.compile'd versions in one Python namespace so the only
# difference between them is the `torch.compile` call. Each call goes through
# exactly one PythonCall round-trip, so wall-clock differences reflect what's
# actually happening on the GPU rather than per-op FFI cost.
const _grad_fn_eager, _grad_fn_compiled = let
    # `import torch` inside _eager so it lands in the function's __globals__
    # at call time — @pyexec runs with separate globals/locals dicts, and a
    # top-level `import torch` would only populate locals, leaving _eager
    # unable to resolve `torch` when invoked later.
    nt = @pyexec """
import torch
def _eager(W1, W2, X, y):
    import torch
    y1 = torch.tanh(torch.matmul(W1, X))
    diff = torch.matmul(W2, y1) - y
    loss = (diff * diff).sum() / y.shape[1]
    return torch.autograd.grad(loss, W1)[0]
# mode="default" — change to "reduce-overhead" for CUDA Graphs, or
# "max-autotune" for an aggressive autotune pass.
_compiled = torch.compile(_eager)
""" => (_eager::Py, _compiled::Py)
    (nt._eager, nt._compiled)
end

pytorch_grad_eager(W1t, W2t, Xt, yt)    = _grad_fn_eager(W1t, W2t, Xt, yt)
pytorch_grad_compiled(W1t, W2t, Xt, yt) = _grad_fn_compiled(W1t, W2t, Xt, yt)

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

# CUDATools.Profile.ProfileResults' default `show` calls `format_bytes` on a
# column that can contain `Inf` (e.g. for non-memcpy kernels), which throws
# `InexactError(Int64, Inf)`. Walk `trace.device` directly to sidestep it.
function summarize_julia_trace(io::IO, trace)
    dev = trace.device
    names_ = dev.name
    starts = dev.start
    stops  = dev.stop
    counts = Dict{String,Int}()
    totals = Dict{String,Float64}()  # in seconds
    order  = String[]
    for i in eachindex(names_)
        nm = String(names_[i])
        if !haskey(counts, nm)
            push!(order, nm)
            counts[nm] = 0
            totals[nm] = 0.0
        end
        counts[nm] += 1
        totals[nm] += stops[i] - starts[i]
    end
    sorted = sort(order; by = nm -> -totals[nm])
    @printf io "  %-66s %6s %10s\n" "kernel" "count" "total µs"
    println(io, "  ", "-"^66, " ", "-"^6, " ", "-"^10)
    for nm in sorted
        label = length(nm) <= 66 ? nm : first(nm, 63) * "..."
        @printf io "  %-66s %6d %10.2f\n" label counts[nm] 1e6 * totals[nm]
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

const _pygc = pyimport("gc")

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
    grad_pytorch_eager = torch_to_julia(pytorch_grad_eager(W1t, W2t, Xt, yt))
    torch.cuda.synchronize()

    # First call to the compiled fn for this shape triggers Inductor codegen
    # (can take seconds). Time it so the user knows.
    print("torch.compile codegen for h=$h ... "); flush(stdout)
    t_compile = @elapsed begin
        pytorch_grad_compiled(W1t, W2t, Xt, yt)
        torch.cuda.synchronize()
    end
    @printf "%.2f s\n" t_compile
    grad_pytorch_compiled = torch_to_julia(pytorch_grad_compiled(W1t, W2t, Xt, yt))
    torch.cuda.synchronize()

    # ----- Numerical equivalence -----
    for (name, g) in [("eager   ", grad_pytorch_eager),
                      ("compiled", grad_pytorch_compiled)]
        maxdiff = maximum(abs.(grad_julia .- g))
        relmag  = maxdiff / max(maximum(abs.(grad_julia)), eps(Float32))
        ok      = isapprox(grad_julia, g; rtol = rtol, atol = 1f-4)
        @printf "PyTorch %s vs Julia:  max|Δ| = %.3e (rel %.2e)  match=%s\n" name maxdiff relmag ok
    end

    # ----- Benchmarks -----
    # samples=30 evals=1 caps total iterations so the PyTorch caching allocator
    # doesn't blow up at h=4096; setup= clears it between samples.
    println("\n--- benchmark (median of 30 samples, post-sync) ---")
    bj = @benchmark begin
        reverse_diff($W1g, $W2g, $Xg, $yg)
        CUDA.synchronize()
    end samples=30 evals=1 seconds=10
    be = @benchmark begin
        pytorch_grad_eager($W1t, $W2t, $Xt, $yt)
        $torch.cuda.synchronize()
    end samples=30 evals=1 seconds=10 setup=($(_pygc).collect(); $torch.cuda.empty_cache())
    bc = @benchmark begin
        pytorch_grad_compiled($W1t, $W2t, $Xt, $yt)
        $torch.cuda.synchronize()
    end samples=30 evals=1 seconds=10 setup=($(_pygc).collect(); $torch.cuda.empty_cache())
    @printf "Julia (CUDA.jl)  : median %8.3f µs\n" 1e-3 * median(bj).time
    @printf "PyTorch eager    : median %8.3f µs\n" 1e-3 * median(be).time
    @printf "PyTorch compiled : median %8.3f µs\n" 1e-3 * median(bc).time

    # ----- CUDA traces -----
    println("\n--- CUDA trace: Julia / CUDA.jl ---")
    summarize_julia_trace(stdout, julia_trace(() -> reverse_diff(W1g, W2g, Xg, yg)))

    println("\n--- CUDA trace: PyTorch eager ---")
    println(pytorch_trace(() -> pytorch_grad_eager(W1t, W2t, Xt, yt)))

    println("\n--- CUDA trace: PyTorch compiled ---")
    println(pytorch_trace(() -> pytorch_grad_compiled(W1t, W2t, Xt, yt)))

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
        # Release per-h tensors from both caching allocators before the next sweep.
        GC.gc(true)
        CUDA.reclaim()
        _pygc.collect()
        torch.cuda.empty_cache()
    end
end

main()
