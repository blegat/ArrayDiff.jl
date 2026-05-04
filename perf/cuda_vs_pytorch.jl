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
using CUDA: AS
using BenchmarkTools
using PythonCall
using Lux
import Mooncake

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
# Hardcoded CUDA.jl path — vectorized custom-kernel version
#
# Replaces the two big elementwise broadcasts:
#   1. tanh.(W1 * X) and 1 .- y_1.^2  →  one fused @cuda kernel
#   2. J_1 .* (W2' * J_2)             →  one @cuda kernel
# with kernels that issue `ld.global.v4.f32` / `st.global.v4.f32` PTX,
# matching PyTorch's `vectorized_elementwise_kernel<4, ...>` shape.
#
# The third broadcast (J_2 from a 1×n vector) is left as a regular .= since
# n is tiny (e.g. 178) and vectorization wouldn't measurably help.
#
# Pattern is taken from CUDA.jl's own ldg.jl tests, which assert that
# NTuple{4, Base.VecElement{Float32}} loads via Core.LLVMPtr lower to
# `ld.global.v4` PTX.
# -------------------------------------------------------------------------
const Float4 = NTuple{4, Base.VecElement{Float32}}

@inline _f4ptr(arr::CuDeviceArray{Float32}) =
    reinterpret(Core.LLVMPtr{Float4, AS.Global}, pointer(arr))

@inline _vec4(t1, t2, t3, t4) =
    (Base.VecElement(t1), Base.VecElement(t2), Base.VecElement(t3), Base.VecElement(t4))

function _tanh_and_jac_kernel!(y, J, x, n::Int)
    i    = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    base = 4 * (i - 1)
    if base + 4 <= n
        v  = unsafe_load(_f4ptr(x), i, Val(16))
        t1 = tanh(v[1].value); t2 = tanh(v[2].value)
        t3 = tanh(v[3].value); t4 = tanh(v[4].value)
        unsafe_store!(_f4ptr(y), _vec4(t1, t2, t3, t4), i, Val(16))
        unsafe_store!(_f4ptr(J),
            _vec4(1f0 - t1*t1, 1f0 - t2*t2, 1f0 - t3*t3, 1f0 - t4*t4), i, Val(16))
    elseif base < n
        for k in 1:(n - base)
            @inbounds t = tanh(x[base + k])
            @inbounds y[base + k] = t
            @inbounds J[base + k] = 1f0 - t*t
        end
    end
    return nothing
end

function tanh_and_jac!(y::CuArray{Float32}, J::CuArray{Float32}, x::CuArray{Float32})
    n       = length(x)
    threads = 256
    blocks  = cld(cld(n, 4), threads)
    @cuda threads=threads blocks=blocks _tanh_and_jac_kernel!(y, J, x, n)
    return (y, J)
end

function _vmul_kernel!(out, a, b, n::Int)
    i    = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    base = 4 * (i - 1)
    if base + 4 <= n
        va = unsafe_load(_f4ptr(a), i, Val(16))
        vb = unsafe_load(_f4ptr(b), i, Val(16))
        unsafe_store!(_f4ptr(out),
            _vec4(va[1].value*vb[1].value, va[2].value*vb[2].value,
                  va[3].value*vb[3].value, va[4].value*vb[4].value), i, Val(16))
    elseif base < n
        for k in 1:(n - base)
            @inbounds out[base + k] = a[base + k] * b[base + k]
        end
    end
    return nothing
end

function vmul!(out::CuArray{Float32}, a::CuArray{Float32}, b::CuArray{Float32})
    n       = length(a)
    threads = 256
    blocks  = cld(cld(n, 4), threads)
    @cuda threads=threads blocks=blocks _vmul_kernel!(out, a, b, n)
    return out
end

function reverse_diff_v4(W1, W2, X, y)
    Z1  = W1 * X                                 # GEMM (h, n)
    y_1 = similar(Z1)
    J_1 = similar(Z1)
    tanh_and_jac!(y_1, J_1, Z1)                  # fused tanh + (1 - y²), vec=4
    J_2 = 2 .* (W2 * y_1 .- y) ./ size(y, 2)     # 1×n broadcast, kept as-is
    tmp = W2' * J_2                              # GEMM (h, n)
    out = similar(tmp)
    vmul!(out, J_1, tmp)                         # J_1 .* tmp, vec=4
    return out * X'                              # GEMM (h, d)
end

# -------------------------------------------------------------------------
# v5: vec=4 elementwise + SIMT cuBLAS GEMM (no TF32 tensor cores)
#
# Under FAST_MATH, gemmExComputeType picks CUBLAS_COMPUTE_32F_FAST_TF32, which
# routes Float32 matmul through TF32 tensor cores. For our awkward k dims
# (k=13 for W1*X, k=178 for out*X', k=1 for W2'*J_2), tensor cores can't be
# filled efficiently and the resulting cutlass_80_tensorop kernel runs much
# slower than a SIMT FP32 GEMM. We bypass gemmExComputeType by calling
# cublasGemmEx directly with CUBLAS_COMPUTE_32F + CUBLAS_GEMM_DEFAULT.
# -------------------------------------------------------------------------
function _gemm_simt!(C::CuArray{Float32,2}, transA::Char, A::CuArray{Float32,2},
                     transB::Char, B::CuArray{Float32,2};
                     alpha::Float32 = 1f0, beta::Float32 = 0f0)
    m   = size(A, transA == 'N' ? 1 : 2)
    k   = size(A, transA == 'N' ? 2 : 1)
    n   = size(B, transB == 'N' ? 2 : 1)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    ldc = max(1, stride(C, 2))
    # CUDA.jl puts the cuBLAS handle in CUBLAS_POINTER_MODE_DEVICE, so alpha/beta
    # MUST be device pointers (host Ref triggers UVA fault handling — 100× slowdown).
    α = CUDA.CuRef{Float32}(alpha); β = CUDA.CuRef{Float32}(beta)
    h = CUDA.CUBLAS.handle()
    # Under FAST_MATH the handle's math mode is CUBLAS_TF32_TENSOR_OP_MATH, which
    # forces TF32 tensor cores even when we ask for CUBLAS_COMPUTE_32F. Flip it to
    # DEFAULT_MATH for this call so cuBLAS picks a SIMT FP32 kernel.
    CUDA.CUBLAS.math_mode!(h, CUDA.DEFAULT_MATH)
    try
        CUDA.CUBLAS.cublasGemmEx(
            h, transA, transB, m, n, k,
            α, A, Float32, lda,
            B, Float32, ldb,
            β, C, Float32, ldc,
            CUDA.CUBLAS.CUBLAS_COMPUTE_32F,
            CUDA.CUBLAS.CUBLAS_GEMM_DEFAULT,
        )
    finally
        CUDA.CUBLAS.math_mode!(h, CUDA.math_mode())  # restore (FAST_MATH → TF32 tensor op)
    end
    return C
end

function reverse_diff_v5(W1, W2, X, y)
    h, d = size(W1)
    nn   = size(X, 2)

    Z1 = CuArray{Float32}(undef, h, nn)
    _gemm_simt!(Z1, 'N', W1, 'N', X)              # SIMT: (h,d) * (d,n)

    y_1 = similar(Z1)
    J_1 = similar(Z1)
    tanh_and_jac!(y_1, J_1, Z1)

    J_2 = 2 .* (W2 * y_1 .- y) ./ size(y, 2)      # tiny, leave as broadcast

    tmp = CuArray{Float32}(undef, h, nn)
    _gemm_simt!(tmp, 'T', W2, 'N', J_2)           # SIMT: W2' * J_2  (k=1)

    out = similar(tmp)
    vmul!(out, J_1, tmp)

    result = CuArray{Float32}(undef, h, d)
    _gemm_simt!(result, 'N', out, 'T', X)         # SIMT: out * X'   (k=n)
    return result
end

# -------------------------------------------------------------------------
# v6: vec=4 elementwise + cuBLASLt with per-shape heuristic-picked algo
#
# cuBLAS's standard heuristic, even with CUBLAS_COMPUTE_32F + DEFAULT_MATH,
# picks `cutlass_80_simt_sgemm_*` for our awkward shapes. PyTorch's process
# happens to land on `magma_sgemmEx_kernel` for the same compute type — same
# library, different choice. cuBLASLt exposes a fuller heuristic API with a
# workspace budget that often unlocks better algos. We build a matmul
# descriptor + layouts per (transA, transB, m, n, k) shape, ask cuBLASLt for
# its best algo, and reuse the cached plan on every call.
# -------------------------------------------------------------------------
const _LT_WS_BYTES = Csize_t(32 * 1024 * 1024)        # 32 MiB workspace

# Lazy: created on first use, kept alive for the process.
const _LT_STATE = Ref{Any}(nothing)

function _lt_state()
    s = _LT_STATE[]
    if s === nothing
        h_ref = Ref{CUDA.CUBLAS.cublasLtHandle_t}(C_NULL)
        CUDA.CUBLAS.cublasLtCreate(h_ref)
        ws = CUDA.CuArray{UInt8}(undef, Int(_LT_WS_BYTES))
        s = (handle = h_ref[], ws = ws)
        _LT_STATE[] = s
    end
    return s::NamedTuple{(:handle, :ws)}
end

mutable struct LtPlan
    desc::CUDA.CUBLAS.cublasLtMatmulDesc_t
    Adesc::CUDA.CUBLAS.cublasLtMatrixLayout_t
    Bdesc::CUDA.CUBLAS.cublasLtMatrixLayout_t
    Cdesc::CUDA.CUBLAS.cublasLtMatrixLayout_t
    algo::CUDA.CUBLAS.cublasLtMatmulAlgo_t
end

function _build_lt_plan(transA::Char, transB::Char,
                        m::Int, n::Int, k::Int,
                        lda::Int, ldb::Int, ldc::Int)
    state  = _lt_state()
    handle = state.handle
    R32    = CUDA.CUDACore.R_32F     # cudaDataType for Float32

    desc_ref = Ref{CUDA.CUBLAS.cublasLtMatmulDesc_t}(C_NULL)
    CUDA.CUBLAS.cublasLtMatmulDescCreate(desc_ref, CUDA.CUBLAS.CUBLAS_COMPUTE_32F, R32)
    desc = desc_ref[]

    # Set transpose attributes.
    tA = (transA == 'N') ? CUDA.CUBLAS.CUBLAS_OP_N : CUDA.CUBLAS.CUBLAS_OP_T
    tB = (transB == 'N') ? CUDA.CUBLAS.CUBLAS_OP_N : CUDA.CUBLAS.CUBLAS_OP_T
    let r = Ref(tA)
        CUDA.CUBLAS.cublasLtMatmulDescSetAttribute(
            desc, CUDA.CUBLAS.CUBLASLT_MATMUL_DESC_TRANSA, r, sizeof(tA))
    end
    let r = Ref(tB)
        CUDA.CUBLAS.cublasLtMatmulDescSetAttribute(
            desc, CUDA.CUBLAS.CUBLASLT_MATMUL_DESC_TRANSB, r, sizeof(tB))
    end

    # Layout shape is the *storage* shape (pre-transpose).
    Arows = transA == 'N' ? m : k
    Acols = transA == 'N' ? k : m
    Brows = transB == 'N' ? k : n
    Bcols = transB == 'N' ? n : k

    Aref = Ref{CUDA.CUBLAS.cublasLtMatrixLayout_t}(C_NULL)
    Bref = Ref{CUDA.CUBLAS.cublasLtMatrixLayout_t}(C_NULL)
    Cref = Ref{CUDA.CUBLAS.cublasLtMatrixLayout_t}(C_NULL)
    CUDA.CUBLAS.cublasLtMatrixLayoutCreate(Aref, R32, UInt64(Arows), UInt64(Acols), Int64(lda))
    CUDA.CUBLAS.cublasLtMatrixLayoutCreate(Bref, R32, UInt64(Brows), UInt64(Bcols), Int64(ldb))
    CUDA.CUBLAS.cublasLtMatrixLayoutCreate(Cref, R32, UInt64(m),     UInt64(n),     Int64(ldc))

    # Preference: tell the heuristic how much workspace it can use.
    pref_ref = Ref{CUDA.CUBLAS.cublasLtMatmulPreference_t}(C_NULL)
    CUDA.CUBLAS.cublasLtMatmulPreferenceCreate(pref_ref)
    pref = pref_ref[]
    let r = Ref(_LT_WS_BYTES)
        CUDA.CUBLAS.cublasLtMatmulPreferenceSetAttribute(
            pref, CUDA.CUBLAS.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            r, sizeof(_LT_WS_BYTES))
    end

    # Heuristic: top-1 algorithm.
    heur     = Vector{CUDA.CUBLAS.cublasLtMatmulHeuristicResult_t}(undef, 1)
    returned = Ref{Cint}(0)
    CUDA.CUBLAS.cublasLtMatmulAlgoGetHeuristic(
        handle, desc, Aref[], Bref[], Cref[], Cref[],
        pref, Cint(1), heur, returned)
    returned[] < 1 && error("cuBLASLt has no algo for shape (m=$m,n=$n,k=$k,trans=$transA$transB)")

    return LtPlan(desc, Aref[], Bref[], Cref[], heur[1].algo)
end

function _gemm_lt!(plan::LtPlan,
                   C::CuArray{Float32,2}, A::CuArray{Float32,2}, B::CuArray{Float32,2};
                   alpha::Float32 = 1f0, beta::Float32 = 0f0)
    state    = _lt_state()
    # cuBLASLt's matmul descriptor defaults to CUBLASLT_POINTER_MODE_HOST
    # (independent of the cuBLAS handle's pointer mode), so alpha/beta are
    # plain host Refs here — using CuRef would trigger UVA faults.
    α        = Ref{Float32}(alpha)
    β        = Ref{Float32}(beta)
    algo_ref = Ref(plan.algo)
    CUDA.CUBLAS.cublasLtMatmul(
        state.handle, plan.desc,
        α, A, plan.Adesc,
           B, plan.Bdesc,
        β, C, plan.Cdesc,
           C, plan.Cdesc,                # D = C in place
        algo_ref,
        state.ws, sizeof(state.ws),
        CUDA.stream(),
    )
    return C
end

# Three plans for our specific 2-layer MLP shape.
struct LtPlans
    p1::LtPlan   # W1 * X      :  (h,d) * (d,n)  → (h,n)
    p2::LtPlan   # W2' * J_2   :  store (1,h),'T' * (1,n)  → (h,n)
    p3::LtPlan   # out * X'    :  (h,n) * store (d,n),'T'  → (h,d)
end

function build_lt_plans(W1::CuArray{Float32,2}, W2::CuArray{Float32,2},
                        X::CuArray{Float32,2})
    h, d = size(W1)
    nn   = size(X, 2)
    p1 = _build_lt_plan('N', 'N', h, nn, d, h, d, h)
    p2 = _build_lt_plan('T', 'N', h, nn, 1, 1, 1, h)
    p3 = _build_lt_plan('N', 'T', h, d, nn, h, d, h)
    return LtPlans(p1, p2, p3)
end

function reverse_diff_v6(plans::LtPlans, W1, W2, X, y)
    h, d = size(W1)
    nn   = size(X, 2)

    Z1 = CuArray{Float32}(undef, h, nn)
    _gemm_lt!(plans.p1, Z1, W1, X)

    y_1 = similar(Z1)
    J_1 = similar(Z1)
    tanh_and_jac!(y_1, J_1, Z1)

    J_2 = 2 .* (W2 * y_1 .- y) ./ size(y, 2)

    tmp = CuArray{Float32}(undef, h, nn)
    _gemm_lt!(plans.p2, tmp, W2, J_2)

    out = similar(tmp)
    vmul!(out, J_1, tmp)

    result = CuArray{Float32}(undef, h, d)
    _gemm_lt!(plans.p3, result, out, X)
    return result
end

# -------------------------------------------------------------------------
# Lux + Mooncake path
#
# Builds an equivalent 2-layer MLP `Y = W2 * tanh(W1 * X)` (no bias) using
# Lux, plugs in the *same* CuArray weights so the gradient is comparable,
# and uses Mooncake (the modern Julia 1.12-friendly reverse-mode AD) for the
# backward. Goes through the same CUDA.jl + cuBLAS stack as `reverse_diff`,
# so kernels should look similar — what we're measuring is the AD/dispatch
# overhead Lux+Mooncake add on top.
# -------------------------------------------------------------------------
struct LuxMooncake{M,P,S,L,R}
    model::M
    ps::P
    st::S
    loss_fn::L
    rule::R
end

function build_lux(W1g::CuArray{Float32,2}, W2g::CuArray{Float32,2},
                   Xg::CuArray, yg::CuArray)
    h, d  = size(W1g)
    model = Lux.Chain(
        Lux.Dense(d => h, tanh; use_bias = false),
        Lux.Dense(h => 1, identity; use_bias = false),
    )
    ps = (
        layer_1 = (weight = W1g,),
        layer_2 = (weight = W2g,),
    )
    st = Lux.initialstates(Random.default_rng(), model)

    # Closure captures Xg, yg, model, st — only `p` is the differentiated arg.
    loss_fn = let model = model, st = st, Xg = Xg, yg = yg
        p -> begin
            y_hat, _ = model(Xg, p, st)
            return sum((y_hat .- yg) .^ 2) / size(yg, 2)
        end
    end

    # build_rrule is the expensive step (compiles the reverse pass for these
    # types) — do it once at setup so the per-call cost in the benchmark is
    # just the actual fwd+bwd execution.
    rule = Mooncake.build_rrule(loss_fn, ps)
    return LuxMooncake(model, ps, st, loss_fn, rule)
end

function lux_grad(lm::LuxMooncake)
    _, (_, ∂ps) = Mooncake.value_and_gradient!!(lm.rule, lm.loss_fn, lm.ps)
    return ∂ps.layer_1.weight
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
    grad_julia    = Array(reverse_diff(W1g, W2g, Xg, yg))
    grad_julia_v4 = Array(reverse_diff_v4(W1g, W2g, Xg, yg))
    grad_julia_v5 = Array(reverse_diff_v5(W1g, W2g, Xg, yg))
    print("cuBLASLt build_lt_plans for h=$h ... "); flush(stdout)
    t_lt_build = @elapsed lt_plans = build_lt_plans(W1g, W2g, Xg)
    @printf "%.3f s\n" t_lt_build
    grad_julia_v6 = Array(reverse_diff_v6(lt_plans, W1g, W2g, Xg, yg))
    CUDA.synchronize()

    # Lux + Mooncake setup. build_rrule compiles the reverse pass for these
    # types (one-time cost per shape); first call afterwards still does some
    # JIT, so we time both separately.
    print("Lux+Mooncake build_rrule for h=$h ... "); flush(stdout)
    t_lux_build = @elapsed lm = build_lux(W1g, W2g, Xg, yg)
    @printf "%.2f s, " t_lux_build
    print("first call ... "); flush(stdout)
    t_lux_first = @elapsed begin
        lux_grad(lm); CUDA.synchronize()
    end
    @printf "%.2f s\n" t_lux_first
    grad_lux = Array(lux_grad(lm))
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
    for (name, g) in [("Julia v4 (vec=4)   ", grad_julia_v4),
                      ("Julia v5 (vec=4+SIMT)", grad_julia_v5),
                      ("Julia v6 (vec=4+Lt)", grad_julia_v6),
                      ("Lux + Mooncake     ", grad_lux),
                      ("PyTorch eager      ", grad_pytorch_eager),
                      ("PyTorch compiled   ", grad_pytorch_compiled)]
        maxdiff = maximum(abs.(grad_julia .- g))
        relmag  = maxdiff / max(maximum(abs.(grad_julia)), eps(Float32))
        ok      = isapprox(grad_julia, g; rtol = rtol, atol = 1f-4)
        @printf "%s vs Julia broadcast: max|Δ| = %.3e (rel %.2e)  match=%s\n" name maxdiff relmag ok
    end

    # ----- Benchmarks -----
    # samples=30 evals=1 caps total iterations so the PyTorch caching allocator
    # doesn't blow up at h=4096; setup= clears it between samples.
    println("\n--- benchmark (median of 30 samples, post-sync) ---")
    bj = @benchmark begin
        reverse_diff($W1g, $W2g, $Xg, $yg)
        CUDA.synchronize()
    end samples=30 evals=1 seconds=10
    bj4 = @benchmark begin
        reverse_diff_v4($W1g, $W2g, $Xg, $yg)
        CUDA.synchronize()
    end samples=30 evals=1 seconds=10
    bj5 = @benchmark begin
        reverse_diff_v5($W1g, $W2g, $Xg, $yg)
        CUDA.synchronize()
    end samples=30 evals=1 seconds=10
    bj6 = @benchmark begin
        reverse_diff_v6($lt_plans, $W1g, $W2g, $Xg, $yg)
        CUDA.synchronize()
    end samples=30 evals=1 seconds=10
    bjlux = @benchmark begin
        lux_grad($lm)
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
    @printf "Julia broadcast      : median %8.3f µs\n" 1e-3 * median(bj).time
    @printf "Julia vec=4          : median %8.3f µs\n" 1e-3 * median(bj4).time
    @printf "Julia vec=4 + SIMT   : median %8.3f µs\n" 1e-3 * median(bj5).time
    @printf "Julia vec=4 + cuBLASLt: median %8.3f µs\n" 1e-3 * median(bj6).time
    @printf "Lux + Mooncake       : median %8.3f µs\n" 1e-3 * median(bjlux).time
    @printf "PyTorch eager        : median %8.3f µs\n" 1e-3 * median(be).time
    @printf "PyTorch compiled     : median %8.3f µs\n" 1e-3 * median(bc).time

    # ----- CUDA traces -----
    println("\n--- CUDA trace: Julia broadcast ---")
    summarize_julia_trace(stdout, julia_trace(() -> reverse_diff(W1g, W2g, Xg, yg)))

    println("\n--- CUDA trace: Julia vec=4 ---")
    summarize_julia_trace(stdout, julia_trace(() -> reverse_diff_v4(W1g, W2g, Xg, yg)))

    println("\n--- CUDA trace: Julia vec=4 + SIMT ---")
    summarize_julia_trace(stdout, julia_trace(() -> reverse_diff_v5(W1g, W2g, Xg, yg)))

    println("\n--- CUDA trace: Julia vec=4 + cuBLASLt ---")
    summarize_julia_trace(stdout, julia_trace(() -> reverse_diff_v6(lt_plans, W1g, W2g, Xg, yg)))

    println("\n--- CUDA trace: Lux + Mooncake ---")
    summarize_julia_trace(stdout, julia_trace(() -> lux_grad(lm)))

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
    # Match PyTorch's default of fast tanh / fast intrinsics. Affects BOTH
    # Julia versions equally, so the broadcast-vs-vec=4 comparison still
    # isolates the kernel-design effect.
    CUDA.math_mode!(CUDA.FAST_MATH)
    println("CUDA.jl device : ", CUDA.name(CUDA.device()), "  (math_mode=FAST_MATH)")
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
