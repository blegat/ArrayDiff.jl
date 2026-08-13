# Second-order (Ipopt) reference solutions used to validate the first-order
# GPU-style solves.

import Ipopt
import JuMP
import PowerModels

"""
    rect_reference(d::RectData) -> objective

Solve the scalar version of form 1 (the JuMP tutorial's model, in per-unit)
with Ipopt. For `case9mod()` this reproduces the tutorial's 3087.84.
"""
function rect_reference(d::RectData)
    N = d.N
    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, -d.vmax <= Vr[1:N] <= d.vmax, start = 1.0)
    JuMP.@variable(model, -d.vmax <= Vi[1:N] <= d.vmax, start = 0.0)
    JuMP.@variable(model, d.Pg_lb[i] <= Pg[i in 1:N] <= d.Pg_ub[i])
    JuMP.@variable(model, d.Qg_lb[i] <= Qg[i in 1:N] <= d.Qg_ub[i])
    JuMP.set_lower_bound(Vr[1], 0.0)
    JuMP.set_lower_bound(Vi[1], 0.0)
    JuMP.set_upper_bound(Vi[1], 0.0)
    G, B = Matrix(d.G), Matrix(d.B)
    JuMP.@expression(
        model,
        Ir[i = 1:N],
        sum(G[i, j] * Vr[j] - B[i, j] * Vi[j] for j in 1:N)
    )
    JuMP.@expression(
        model,
        Ii[i = 1:N],
        sum(G[i, j] * Vi[j] + B[i, j] * Vr[j] for j in 1:N)
    )
    JuMP.@constraint(
        model,
        [i = 1:N],
        Pg[i] - d.Pd[i] == Vr[i] * Ir[i] + Vi[i] * Ii[i]
    )
    JuMP.@constraint(
        model,
        [i = 1:N],
        Qg[i] - d.Qd[i] == Vi[i] * Ir[i] - Vr[i] * Ii[i]
    )
    JuMP.@constraint(model, [i = 1:N], Vr[i]^2 + Vi[i]^2 >= d.vmin^2)
    JuMP.@constraint(model, [i = 1:N], Vr[i]^2 + Vi[i]^2 <= d.vmax^2)
    JuMP.@objective(
        model,
        Min,
        sum(d.c2[i] * Pg[i]^2 + d.c1[i] * Pg[i] for i in 1:N) + d.c0
    )
    JuMP.optimize!(model)
    return JuMP.objective_value(model)
end

"""
    polar_reference(file) -> objective

PowerModels' AC-OPF (polar) solved with Ipopt. Note: PowerModels includes
angle-difference bounds that the GenOpt/ExaModels form omits; they are
inactive at the optimum for the standard test cases used here.
"""
function polar_reference(file::AbstractString)
    PowerModels.silence()
    result = PowerModels.solve_ac_opf(
        file,
        JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
    )
    return result["objective"]
end
