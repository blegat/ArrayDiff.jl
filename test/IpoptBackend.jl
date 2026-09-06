module TestIpoptBackend

using Test
import ArrayDiff
import Ipopt
import MathOptInterface as MOI

function _square_minus(x, value)
    difference = MOI.ScalarNonlinearFunction(:-, Any[x, value])
    return MOI.ScalarNonlinearFunction(:^, Any[difference, 2.0])
end

function test_ipopt_with_arraydiff_mode()
    model = Ipopt.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    mode = ArrayDiff.Mode()
    MOI.set(model, MOI.AutomaticDifferentiationBackend(), mode)
    @test MOI.get(model, MOI.AutomaticDifferentiationBackend()) === mode

    x, y = MOI.add_variables(model, 2)
    objective = MOI.ScalarNonlinearFunction(
        :+,
        Any[_square_minus(x, 1.0), _square_minus(y, 2.0)],
    )
    MOI.set(model, MOI.ObjectiveFunction{typeof(objective)}(), objective)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    quadratic = MOI.ScalarQuadraticFunction(
        [
            MOI.ScalarQuadraticTerm(2.0, x, x),
            MOI.ScalarQuadraticTerm(2.0, y, y),
        ],
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    MOI.add_constraint(model, quadratic, MOI.LessThan(10.0))

    oracle = MOI.VectorNonlinearOracle(;
        dimension = 2,
        l = [3.0],
        u = [Inf],
        eval_f = (output, input) -> (output[1] = input[1] + input[2]),
        jacobian_structure = [(1, 1), (1, 2)],
        eval_jacobian = (values, input) -> (values .= 1.0),
        hessian_lagrangian_structure = Tuple{Int,Int}[],
        eval_hessian_lagrangian = (values, input, multipliers) -> nothing,
    )
    MOI.add_constraint(model, MOI.VectorOfVariables([x, y]), oracle)

    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 1.0 atol = 1e-4
    @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 2.0 atol = 1e-4
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 0.0 atol = 1e-8
    return
end

test_ipopt_with_arraydiff_mode()

end  # module
