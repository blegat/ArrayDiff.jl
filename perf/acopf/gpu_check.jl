# GPU-semantics validation without a GPU: run both forms with the tape and
# all solver state on `JLArrays.JLArray` (the GPUArrays.jl reference backend)
# with scalar indexing disallowed. Any operation that would break on CUDA
# (scalar getindex/setindex on device arrays) throws here.
#
# Evaluations (residuals, J'v, objective gradient) are compared point-wise
# against the CPU path at the same input — these are deterministic up to
# reduction order, so tolerances are tight. Full solves are only smoke-tested
# (thousands of Adam steps on a nonconvex problem amplify last-bit
# differences between CPU and GPU-style reductions, so trajectories are not
# bitwise comparable).
#
#   julia --project=. gpu_check.jl

include("ACOPF.jl")

import GPUArraysCore
import JLArrays
import Random
import Test

GPUArraysCore.allowscalar(false)

const JLV = JLArrays.JLArray{Float64,1}

jl_device(x::AbstractArray) = JLArrays.JLArray(x)

function compare_evaluations(name, prob_cpu, prob_dev)
    println("── $name: evaluation comparison")
    Random.seed!(42)
    x_cpu = clamp.(Vector(prob_cpu.x0) .+ 0.01 .* randn(length(prob_cpu.x0)),
        Vector(prob_cpu.lb), Vector(prob_cpu.ub))
    x_dev = JLV(x_cpu)
    g_cpu = zero(x_cpu)
    g_dev = JLV(zero(x_cpu))
    MOI.eval_objective_gradient(prob_cpu.objective, g_cpu, x_cpu)
    MOI.eval_objective_gradient(prob_dev.objective, g_dev, x_dev)
    Test.@test MOI.eval_objective(prob_dev.objective, x_dev) ≈
               MOI.eval_objective(prob_cpu.objective, x_cpu) rtol = 1e-12
    Test.@test Vector(g_dev) ≈ g_cpu rtol = 1e-12
    for (gc, gd) in zip(prob_cpu.groups, prob_dev.groups)
        F_cpu = zeros(gc.dim)
        F_dev = JLV(zeros(gd.dim))
        ArrayDiff.eval_residual!(gc.evaluator, F_cpu, x_cpu)
        ArrayDiff.eval_residual!(gd.evaluator, F_dev, x_dev)
        Test.@test Vector(F_dev) ≈ F_cpu rtol = 1e-12 atol = 1e-14
        v = randn(gc.dim)
        Jtv_cpu = zero(x_cpu)
        Jtv_dev = JLV(zero(x_cpu))
        ArrayDiff.eval_residual_jtprod!(gc.evaluator, Jtv_cpu, x_cpu, v)
        ArrayDiff.eval_residual_jtprod!(gd.evaluator, Jtv_dev, x_dev, JLV(v))
        Test.@test Vector(Jtv_dev) ≈ Jtv_cpu rtol = 1e-10 atol = 1e-12
        println("   $(gc.name): F and J'v match")
    end
    return
end

function smoke_solve(name, prob_dev)
    x, stats = solve!(prob_dev; outer = 5, inner = 500, polish = 2, verbose = false)
    println("── $name: smoke solve  obj $(stats.obj)  viol $(stats.viol)")
    Test.@test isfinite(stats.obj)
    Test.@test stats.viol < 5e-2
    return
end

Test.@testset "GPU-semantics (JLArrays) checks" begin
    d1 = case9mod()
    rect_cpu = build_rect(d1; matrix = ELLMatrix)
    rect_dev = build_rect(
        d1;
        matrix = ELLMatrix,
        mode = ArrayDiff.Mode{JLV}(),
        device = jl_device,
    )
    compare_evaluations("rect / ELLMatrix", rect_cpu, rect_dev)
    smoke_solve("rect / ELLMatrix", rect_dev)
    d2 = parse_polar_case(matpower_case("case9.m"))
    polar_cpu = build_polar(d2; use_gather = true)
    polar_dev = build_polar(
        d2;
        use_gather = true,
        mode = ArrayDiff.Mode{JLV}(),
        device = jl_device,
    )
    compare_evaluations("polar / GatherMatrix", polar_cpu, polar_dev)
    smoke_solve("polar / GatherMatrix", polar_dev)
end
