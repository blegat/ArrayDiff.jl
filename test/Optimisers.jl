module TestWithOptimisers

using Test

using JuMP
using ArrayDiff
import LinearAlgebra
import MathOptInterface as MOI
import NLopt
import NLPModels
import NLPModelsJuMP
import Optimisers

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

# Simple NLPModels interface for Optimisers.jl: run an Optimisers rule on the
# variable vector of an `AbstractNLPModel` using `obj` and `grad!`.
function optimize_with_optimiser(
    rule::Optimisers.AbstractRule,
    nlp::NLPModels.AbstractNLPModel;
    max_iters::Int = 10_000,
    tol::Real = 1e-6,
)
    x = copy(nlp.meta.x0)
    g = similar(x)
    state = Optimisers.setup(rule, x)
    for iter in 1:max_iters
        NLPModels.grad!(nlp, x, g)
        if LinearAlgebra.norm(g) < tol
            return (
                x = x,
                objective = NLPModels.obj(nlp, x),
                iters = iter,
                status = :first_order,
            )
        end
        state, x = Optimisers.update!(state, x, g)
    end
    return (
        x = x,
        objective = NLPModels.obj(nlp, x),
        iters = max_iters,
        status = :max_iter,
    )
end

function _test_neural_optimisers(with_norm::Bool)
    n = 2
    X = [1.0 0.5; 0.3 0.8]
    target = [0.5 0.2; 0.1 0.7]
    model = direct_model(NLopt.Optimizer())
    set_attribute(model, "algorithm", :LD_LBFGS)
    @variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    # Use distinct starting values to break symmetry
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
    stats = optimize_with_optimiser(
        Optimisers.Adam(0.05),
        nlp;
        max_iters = 20_000,
        tol = 1e-6,
    )
    @test stats.objective < 1e-3
    return
end

function test_neural_optimisers()
    _test_neural_optimisers(true)
    return _test_neural_optimisers(false)
end

end

TestWithOptimisers.runtests()
