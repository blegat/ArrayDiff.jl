# Largely inspired by MathOptInterface/src/Nonlinear/parse.jl
# Most functions have been copy-pasted and slightly modified to adapt to small changes in OperatorRegistry and Model.

function MOI.initialize(evaluator::Evaluator, features::Vector{Symbol})
    start_time = time()
    empty!(evaluator.ordered_constraints)
    evaluator.eval_objective_timer = 0.0
    evaluator.eval_objective_gradient_timer = 0.0
    evaluator.eval_constraint_timer = 0.0
    evaluator.eval_constraint_gradient_timer = 0.0
    evaluator.eval_constraint_jacobian_timer = 0.0
    evaluator.eval_hessian_objective_timer = 0.0
    evaluator.eval_hessian_constraint_timer = 0.0
    evaluator.eval_hessian_lagrangian_timer = 0.0
    append!(evaluator.ordered_constraints, keys(evaluator.model.constraints))
    # Every backend supports :ExprGraph, so don't forward it.
    filter!(f -> f != :ExprGraph, features)
    if evaluator.backend !== nothing
        MOI.initialize(evaluator.backend, features)
    elseif !isempty(features)
        @assert evaluator.backend === nothing  # ==> ExprGraphOnly used
        error(
            "Unable to initialize `Nonlinear.Evaluator` because the " *
            "following features are not supported: $features",
        )
    end
    evaluator.initialize_timer = time() - start_time
    return
end

function MOI.eval_objective(evaluator::Evaluator, x)
    start = time()
    obj = MOI.eval_objective(evaluator.backend, x)
    evaluator.eval_objective_timer += time() - start
    return obj
end

function MOI.eval_objective_gradient(evaluator::Evaluator, g, x)
    start = time()
    MOI.eval_objective_gradient(evaluator.backend, g, x)
    evaluator.eval_objective_gradient_timer += time() - start
    return
end

function MOI.eval_constraint_gradient(evaluator::Evaluator, ∇g, x, i)
    start = time()
    MOI.eval_constraint_gradient(evaluator.backend, ∇g, x, i)
    evaluator.eval_constraint_gradient_timer += time() - start
    return
end

function MOI.constraint_gradient_structure(evaluator::Evaluator, i)
    return MOI.constraint_gradient_structure(evaluator.backend, i)
end

function MOI.eval_constraint(evaluator::Evaluator, g, x)
    start = time()
    MOI.eval_constraint(evaluator.backend, g, x)
    evaluator.eval_constraint_timer += time() - start
    return
end

function MOI.jacobian_structure(evaluator::Evaluator)
    return MOI.jacobian_structure(evaluator.backend)
end

function MOI.eval_constraint_jacobian(evaluator::Evaluator, J, x)
    start = time()
    MOI.eval_constraint_jacobian(evaluator.backend, J, x)
    evaluator.eval_constraint_jacobian_timer += time() - start
    return
end

function MOI.hessian_objective_structure(evaluator::Evaluator)
    return MOI.hessian_objective_structure(evaluator.backend)
end

function MOI.hessian_constraint_structure(evaluator::Evaluator, i)
    return MOI.hessian_constraint_structure(evaluator.backend, i)
end

function MOI.hessian_lagrangian_structure(evaluator::Evaluator)
    return MOI.hessian_lagrangian_structure(evaluator.backend)
end

function MOI.eval_hessian_objective(evaluator::Evaluator, H, x)
    start = time()
    MOI.eval_hessian_objective(evaluator.backend, H, x)
    evaluator.eval_hessian_objective_timer += time() - start
    return
end

function MOI.eval_hessian_constraint(evaluator::Evaluator, H, x, i)
    start = time()
    MOI.eval_hessian_constraint(evaluator.backend, H, x, i)
    evaluator.eval_hessian_constraint_timer += time() - start
    return
end

function MOI.eval_hessian_lagrangian(evaluator::Evaluator, H, x, σ, μ)
    start = time()
    MOI.eval_hessian_lagrangian(evaluator.backend, H, x, σ, μ)
    evaluator.eval_hessian_lagrangian_timer += time() - start
    return
end

function MOI.eval_constraint_jacobian_product(evaluator::Evaluator, y, x, w)
    start = time()
    MOI.eval_constraint_jacobian_product(evaluator.backend, y, x, w)
    evaluator.eval_constraint_jacobian_timer += time() - start
    return
end

function MOI.eval_constraint_jacobian_transpose_product(
    evaluator::Evaluator,
    y,
    x,
    w,
)
    start = time()
    MOI.eval_constraint_jacobian_transpose_product(evaluator.backend, y, x, w)
    evaluator.eval_constraint_jacobian_timer += time() - start
    return
end

function MOI.eval_hessian_lagrangian_product(
    evaluator::Evaluator,
    H,
    x,
    v,
    σ,
    μ,
)
    start = time()
    MOI.eval_hessian_lagrangian_product(evaluator.backend, H, x, v, σ, μ)
    evaluator.eval_hessian_lagrangian_timer += time() - start
    return
end

function eval_univariate_hessian(
    registry::OperatorRegistry,
    id::Integer,
    x::T,
) where {T}
    if id <= registry.univariate_user_operator_start
        ret = Nonlinear._eval_univariate_2nd_deriv(id, x)
        if ret === nothing
            op = registry.univariate_operators[id]
            error("Hessian is not defined for operator $op")
        end
        return ret::T
    end
    offset = id - registry.univariate_user_operator_start
    operator = registry.registered_univariate_operators[offset]
    return eval_univariate_hessian(operator, x)
end