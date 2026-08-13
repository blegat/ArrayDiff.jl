# Solve both AC-OPF forms on an NVIDIA GPU (tape, solver state, and the
# structured constant matrices all live on the device).
#
#   julia --project=. run_gpu.jl [case]      (default: case9.m for polar)
#
# Requires a CUDA-capable machine: `import Pkg; Pkg.add("CUDA")` in this
# project first. The container this code was developed in has no GPU; this
# script was validated indirectly through gpu_check.jl (JLArrays enforce the
# same device-semantics restrictions as CUDA arrays).

include("ACOPF.jl")
include("reference.jl")

import CUDA
import Printf

@assert CUDA.functional() "CUDA is not functional on this machine"

CUDA.allowscalar(false)

const CUV = CUDA.CuVector{Float64}

cu_device(x::AbstractArray{Float64}) = CUDA.CuArray(x)
cu_device(x::AbstractArray{Int}) = CUDA.CuArray(x)
cu_device(x) = x

function bench(f, name)
    f() # warm up / compile
    t = @elapsed f()
    Printf.@printf("%-28s %10.3f s\n", name, t)
    return
end

function main(case = "case9.m")
    println("═"^70)
    println("Form 1 on GPU: rectangular + Ybus (ELLMatrix), case9mod")
    println("═"^70)
    d1 = case9mod()
    prob1 = build_rect(
        d1;
        matrix = ELLMatrix,
        mode = ArrayDiff.Mode{CUV}(),
        device = cu_device,
    )
    x1, stats1 = solve!(prob1)
    ref1 = rect_reference(d1)
    Printf.@printf(
        "GPU objective %.2f   Ipopt %.2f   viol %.2e\n",
        stats1.obj,
        ref1,
        stats1.viol,
    )
    println()
    println("═"^70)
    println("Form 2 on GPU: polar sin/cos (GatherMatrix), $case")
    println("═"^70)
    d2 = parse_polar_case(matpower_case(case))
    prob2 = build_polar(
        d2;
        use_gather = true,
        mode = ArrayDiff.Mode{CUV}(),
        device = cu_device,
    )
    x2, stats2 = solve!(prob2)
    ref2 = polar_reference(matpower_case(case))
    Printf.@printf(
        "GPU objective %.2f   Ipopt %.2f   viol %.2e\n",
        stats2.obj,
        ref2,
        stats2.viol,
    )
    # AL-gradient timing on the polar form: CPU (CSC) vs GPU with the
    # structured GatherMatrix vs GPU with CUSPARSE CSR.
    println()
    println("AL gradient timing (1000 evaluations), polar form:")
    prob_cpu = build_polar(d2; use_gather = false)
    cusparse(x::SparseArrays.SparseMatrixCSC) =
        CUDA.CUSPARSE.CuSparseMatrixCSR(x)
    cusparse(x::AbstractArray) = CUDA.CuArray(x)
    prob_csr = build_polar(
        d2;
        use_gather = false,
        mode = ArrayDiff.Mode{CUV}(),
        device = cusparse,
    )
    st_cpu = ALState(prob_cpu)
    st_gpu = ALState(prob2)
    st_csr = ALState(prob_csr)
    bench(
        () -> foreach(_ -> al_gradient!(st_cpu, prob_cpu, 10.0), 1:1000),
        "CPU (CSC)",
    )
    bench(
        () -> CUDA.@sync(foreach(_ -> al_gradient!(st_gpu, prob2, 10.0), 1:1000)),
        "GPU (GatherMatrix)",
    )
    bench(
        () -> CUDA.@sync(foreach(_ -> al_gradient!(st_csr, prob_csr, 10.0), 1:1000)),
        "GPU (CUSPARSE CSR)",
    )
    return
end

main(length(ARGS) >= 1 ? ARGS[1] : "case9.m")
