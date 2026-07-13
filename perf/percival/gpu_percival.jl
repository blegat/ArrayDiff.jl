# Run the Percival path with the tape + solver state on `JLArray` (GPUArrays
# reference backend) with scalar indexing disallowed — the same device
# semantics as CUDA. Validates the Percival GPU fix (AugLagModel storage type)
# and that TRON + LBFGSModel + AugLagModel are device-generic.

include("build_percival.jl")

import GPUArraysCore
import JLArrays
import NLPModels
import NLPModelsModifiers
import Percival
import Printf

GPUArraysCore.allowscalar(false)
const JLV = JLArrays.JLArray{Float64, 1}
jl_device(x::AbstractArray) = JLArrays.JLArray(x)

function solve_gpu(nlp; kwargs...)
    return Percival.percival(
        nlp;
        subproblem_modifier = NLPModelsModifiers.LBFGSModel,
        kwargs...,
    )
end

# GPU-native path: the SPG subsolver (pure broadcasts) replaces TRON, and the
# AL subproblem is minimized directly (no quasi-Newton wrapper needed since
# SPG is first-order).
function solve_gpu_spg(nlp; kwargs...)
    return Percival.percival(nlp; subsolver = Percival.SPGSubSolver, kwargs...)
end

# Compare a handful of NLPModels evaluations CPU vs JLArray to prove the
# device path is numerically identical before trusting the full solve.
function check_evals(nlp_cpu, nlp_gpu)
    x_cpu = Vector(nlp_cpu.meta.x0) .+ 0.01
    x_gpu = JLV(x_cpu)
    g_cpu = similar(x_cpu)
    g_gpu = JLV(zero(x_cpu))
    NLPModels.grad!(nlp_cpu, x_cpu, g_cpu)
    NLPModels.grad!(nlp_gpu, x_gpu, g_gpu)
    @assert Vector(g_gpu) ≈ g_cpu
    c_cpu = zeros(nlp_cpu.meta.ncon)
    c_gpu = JLV(zeros(nlp_gpu.meta.ncon))
    NLPModels.cons!(nlp_cpu, x_cpu, c_cpu)
    NLPModels.cons!(nlp_gpu, x_gpu, c_gpu)
    @assert Vector(c_gpu) ≈ c_cpu
    v_cpu = ones(nlp_cpu.meta.ncon)
    Jtv_cpu = similar(x_cpu)
    Jtv_gpu = JLV(zero(x_cpu))
    NLPModels.jtprod!(nlp_cpu, x_cpu, v_cpu, Jtv_cpu)
    NLPModels.jtprod!(nlp_gpu, x_gpu, JLV(v_cpu), Jtv_gpu)
    @assert Vector(Jtv_gpu) ≈ Jtv_cpu
    println("   evals CPU vs JLArray match (grad, cons, jtprod)")
    return
end
