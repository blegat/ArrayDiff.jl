# Finite-difference validation of every residual group and the objective
# gradient for both AC-OPF forms, at a random interior point.
#
#   julia --project=. check_derivatives.jl

include("ACOPF.jl")

import Random
import Test

function _rand_point(prob)
    lb = Vector(prob.lb)
    ub = Vector(prob.ub)
    x = similar(lb)
    for i in eachindex(x)
        lo = isfinite(lb[i]) ? lb[i] : -0.5
        hi = isfinite(ub[i]) ? ub[i] : 0.5
        t = 0.3 + 0.4 * rand()
        x[i] = lo + t * (hi - lo)
    end
    return x
end

function check_group(grp, x; h = 1e-6, atol = 1e-5)
    n = length(x)
    m = grp.dim
    F = zeros(m)
    ArrayDiff.eval_residual!(grp.evaluator, F, x)
    v = randn(m)
    Jtv = zeros(n)
    ArrayDiff.eval_residual_jtprod!(grp.evaluator, Jtv, x, v)
    Fp, Fm = zeros(m), zeros(m)
    Jtv_fd = map(1:n) do i
        xp = copy(x)
        xp[i] += h
        xm = copy(x)
        xm[i] -= h
        ArrayDiff.eval_residual!(grp.evaluator, Fp, xp)
        ArrayDiff.eval_residual!(grp.evaluator, Fm, xm)
        return LinearAlgebra.dot(v, (Fp .- Fm) ./ (2h))
    end
    err = maximum(abs, Jtv .- Jtv_fd)
    Test.@test err < atol
    return err
end

function check_objective(prob, x; h = 1e-6, atol = 1e-5)
    g = zero(x)
    MOI.eval_objective_gradient(prob.objective, g, x)
    g_fd = map(eachindex(x)) do i
        xp = copy(x)
        xp[i] += h
        xm = copy(x)
        xm[i] -= h
        return (
            MOI.eval_objective(prob.objective, xp) -
            MOI.eval_objective(prob.objective, xm)
        ) / (2h)
    end
    err = maximum(abs, g .- g_fd)
    Test.@test err < atol
    return err
end

function check_problem(name, prob)
    println("── $name")
    x = _rand_point(prob)
    err = check_objective(prob, x)
    println("   objective gradient: max err $err")
    for grp in prob.groups
        err = check_group(grp, x)
        println("   $(grp.name): max J'v err $err")
    end
    return
end

Random.seed!(1)
Test.@testset "AC-OPF derivative checks" begin
    d1 = case9mod()
    check_problem("rect / ELLMatrix", build_rect(d1; matrix = ELLMatrix))
    check_problem("rect / SparseMatrixCSC", build_rect(d1; matrix = identity))
    check_problem("rect / dense tape", build_rect(d1; matrix = Matrix))
    d2 = parse_polar_case(matpower_case("case9.m"))
    check_problem("polar / GatherMatrix", build_polar(d2; use_gather = true))
    check_problem("polar / SparseMatrixCSC", build_polar(d2; use_gather = false))
end
