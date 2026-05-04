module TestWithNLPModelsJuMP

using Test

using JuMP
using ArrayDiff
import LinearAlgebra
import MathOptInterface as MOI
import NLopt
import NLPModelsJuMP
import NLPModelsIpopt

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function _test_neural_ipopt_nlpmodels(with_norm::Bool)
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    # Build the JuMP model using direct_model on NLopt (which supports
    # ArrayNonlinearFunction) to set up variables and objective.
    model = direct_model(NLopt.Optimizer())
    set_attribute(model, "algorithm", :LD_LBFGS)
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    start_W1 = [0.3 -0.2; 0.1 0.4]
    start_W2 = [-0.1 0.5; 0.2 -0.3]
    for i in 1:n, j in 1:n
        set_start_value(W1[i, j], start_W1[i, j])
        set_start_value(W2[i, j], start_W2[i, j])
    end
    Y = W2 * tanh.(W1 * X)
    if with_norm
        loss = LinearAlgebra.norm(Y .- target)
    else
        loss = sum((Y .- target) .^ 2)
    end
    @objective(model, Min, loss)
    nlp = NLPModelsJuMP.MathOptNLPModel(model; hessian = false)
    stats = NLPModelsIpopt.ipopt(nlp; print_level = 0)
    @test stats.status == :first_order
    @test stats.objective < 1e-6
    return
end

function test_neural_ipopt_nlpmodels()
    _test_neural_ipopt_nlpmodels(true)
    return _test_neural_ipopt_nlpmodels(false)
end

end

TestWithNLPModelsJuMP.runtests()
