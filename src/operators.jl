# Largely inspired by MathOptInterface/src/Nonlinear/operators.jl
# Most functions have been copy-pasted and slightly modified to adapt to small changes in OperatorRegistry and Model.

const DEFAULT_MULTIVARIATE_OPERATORS = [
    :+,
    :-,
    :*,
    :^,
    :/,
    :ifelse,
    :atan,
    :min,
    :max,
    :vect,
    :dot,
    :hcat,
    :vcat,
    :norm,
    :sum,
    :row,
]

function eval_logic_function(
    ::OperatorRegistry,
    op::Symbol,
    lhs::T,
    rhs::T,
)::Bool where {T}
    if op == :&&
        return lhs && rhs
    else
        @assert op == :||
        return lhs || rhs
    end
end

function _generate_eval_univariate()
    exprs = map(Nonlinear.DEFAULT_UNIVARIATE_OPERATORS) do op
        return :(return (value_deriv_and_second($op, x)[1], value_deriv_and_second($op, x)[2]))
    end
    return Nonlinear._create_binary_switch(1:length(exprs), exprs)
end

@eval @inline function _eval_univariate(id, x::T) where {T}
    $(_generate_eval_univariate())
    return error("Invalid id for univariate operator: $id")
end

function eval_multivariate_function(
    registry::OperatorRegistry,
    op::Symbol,
    x::AbstractVector{T},
) where {T}
    if op == :+
        return sum(x; init = zero(T))
    elseif op == :-
        @assert length(x) == 2
        return x[1] - x[2]
    elseif op == :*
        return prod(x; init = one(T))
    elseif op == :^
        @assert length(x) == 2
        # Use _nan_pow here to avoid throwing an error in common situations like
        # (-1.0)^1.5.
        return _nan_pow(x[1], x[2])
    elseif op == :/
        @assert length(x) == 2
        return x[1] / x[2]
    elseif op == :ifelse
        @assert length(x) == 3
        return ifelse(Bool(x[1]), x[2], x[3])
    elseif op == :atan
        @assert length(x) == 2
        return atan(x[1], x[2])
    elseif op == :min
        return minimum(x)
    elseif op == :max
        return maximum(x)
    elseif op == :vect
        return x
    end
    id = registry.multivariate_operator_to_id[op]
    offset = id - registry.multivariate_user_operator_start
    operator = registry.registered_multivariate_operators[offset]
    @assert length(x) == operator.N
    ret = operator.f(x)
    MOI.Nonlinear.check_return_type(T, ret)
    return ret::T
end

function eval_multivariate_hessian(
    registry::OperatorRegistry,
    op::Symbol,
    H,
    x::AbstractVector{T},
) where {T}
    if op in (:+, :-, :ifelse)
        return false
    end
    if op == :*
        # f(x)    = *(x[i] for i in 1:N)
        #
        # ∇fᵢ(x)  = *(x[j] for j in 1:N if i != j)
        #
        # ∇fᵢⱼ(x) = *(x[k] for k in 1:N if i != k & j != k)
        N = length(x)
        if N == 1
            # Hessian is zero
        elseif N == 2
            H[2, 1] = one(T)
        else
            for i in 1:N, j in (i+1):N
                H[j, i] =
                    prod(x[k] for k in 1:N if k != i && k != j; init = one(T))
            end
        end
    elseif op == :^
        # f(x)   = x[1]^x[2]
        #
        # ∇f(x)  = x[2]*x[1]^(x[2]-1)
        #          x[1]^x[2]*log(x[1])
        #
        # ∇²f(x) = x[2]*(x[2]-1)*x[1]^(x[2]-2)
        #          x[1]^(x[2]-1)*(x[2]*log(x[1])+1) x[1]^x[2]*log(x[1])^2
        ln = x[1] > 0 ? log(x[1]) : NaN
        if x[2] == one(T)
            H[2, 1] = _nan_to_zero(ln + one(T))
            H[2, 2] = _nan_to_zero(x[1] * ln^2)
        elseif x[2] == T(2)
            H[1, 1] = T(2)
            H[2, 1] = _nan_to_zero(x[1] * (T(2) * ln + one(T)))
            H[2, 2] = _nan_to_zero(ln^2 * x[1]^2)
        else
            H[1, 1] = _nan_to_zero(x[2] * (x[2] - 1) * _nan_pow(x[1], x[2] - 2))
            H[2, 1] = _nan_to_zero(_nan_pow(x[1], x[2] - 1) * (x[2] * ln + 1))
            H[2, 2] = _nan_to_zero(ln^2 * _nan_pow(x[1], x[2]))
        end
    elseif op == :/
        # f(x)  = x[1]/x[2]
        #
        # ∇f(x) = 1/x[2]
        #         -x[1]/x[2]^2
        #
        # ∇²(x) = 0.0
        #         -1/x[2]^2 2x[1]/x[2]^3
        d = 1 / x[2]^2
        H[2, 1] = -d
        H[2, 2] = 2 * x[1] * d / x[2]
    elseif op == :atan
        # f(x)  = atan(y, x)
        #
        # ∇f(x) = +x/(x^2+y^2)
        #         -y/(x^2+y^2)
        #
        # ∇²(x) = -(2xy)/(x^2+y^2)^2
        #         (y^2-x^2)/(x^2+y^2)^2 (2xy)/(x^2+y^2)^2
        base = (x[1]^2 + x[2]^2)^2
        H[1, 1] = -2 * x[2] * x[1] / base
        H[2, 1] = (x[1]^2 - x[2]^2) / base
        H[2, 2] = 2 * x[2] * x[1] / base
    elseif op == :min
        _, i = findmin(x)
        H[i, i] = one(T)
    elseif op == :max
        _, i = findmax(x)
        H[i, i] = one(T)
    else
        id = registry.multivariate_operator_to_id[op]
        offset = id - registry.multivariate_user_operator_start
        operator = registry.registered_multivariate_operators[offset]
        if operator.∇²f === nothing
            error("Hessian is not defined for operator $op")
        end
        @assert length(x) == operator.N
        operator.∇²f(H, x)
    end
    return true
end

function _validate_register_assumptions(
    f::Function,
    name::Symbol,
    dimension::Integer,
)
    # Assumption 1: check that `f` can be called with `Float64` arguments.
    y = 0.0
    try
        if dimension == 1
            y = f(0.0)
        else
            y = f(zeros(dimension)...)
        end
    catch
        # We hit some other error, perhaps we called a function like log(-1).
        # Ignore for now, and hope that a useful error is shown to the user
        # during the solve.
    end
    if !(y isa Real)
        error(
            "Expected return type of `Float64` from the user-defined " *
            "function :$(name), but got `$(typeof(y))`.",
        )
    end
    # Assumption 2: check that `f` can be differentiated using `ForwardDiff`.
    try
        if dimension == 1
            ForwardDiff.derivative(f, 0.0)
        else
            ForwardDiff.gradient(x -> f(x...), zeros(dimension))
        end
    catch err
        if err isa MethodError
            error(
                "Unable to register the function :$name.\n\n" *
                _FORWARD_DIFF_METHOD_ERROR_HELPER,
            )
        end
        # We hit some other error, perhaps we called a function like log(-1).
        # Ignore for now, and hope that a useful error is shown to the user
        # during the solve.
    end
    return
end

function _checked_derivative(f::F, op::Symbol) where {F}
    return function (x)
        try
            return ForwardDiff.derivative(f, x)
        catch err
            _intercept_ForwardDiff_MethodError(err, op)
        end
    end
end

"""
    check_return_type(::Type{T}, ret::S) where {T,S}

Overload this method for new types `S` to throw an informative error if a
user-defined function returns the type `S` instead of `T`.
"""
check_return_type(::Type{T}, ret::T) where {T} = nothing

function check_return_type(::Type{T}, ret) where {T}
    return error(
        "Expected return type of $T from a user-defined function, but got " *
        "$(typeof(ret)).",
    )
end

struct _UnivariateOperator{F,F′,F′′}
    f::F
    f′::F′
    f′′::F′′
    function _UnivariateOperator(
        f::Function,
        f′::Function,
        f′′::Union{Nothing,Function} = nothing,
    )
        return new{typeof(f),typeof(f′),typeof(f′′)}(f, f′, f′′)
    end
end

function _UnivariateOperator(op::Symbol, f::Function)
    _validate_register_assumptions(f, op, 1)
    f′ = _checked_derivative(f, op)
    return _UnivariateOperator(op, f, f′)
end

function _UnivariateOperator(op::Symbol, f::Function, f′::Function)
    try
        _validate_register_assumptions(f′, op, 1)
        f′′ = _checked_derivative(f′, op)
        return _UnivariateOperator(f, f′, f′′)
    catch
        return _UnivariateOperator(f, f′, nothing)
    end
end

function _UnivariateOperator(::Symbol, f::Function, f′::Function, f′′::Function)
    return _UnivariateOperator(f, f′, f′′)
end

function eval_univariate_function(operator::_UnivariateOperator, x::T) where {T}
    ret = operator.f(x)
    check_return_type(T, ret)
    return ret::T
end

function eval_univariate_gradient(operator::_UnivariateOperator, x::T) where {T}
    ret = operator.f′(x)
    check_return_type(T, ret)
    return ret::T
end

function eval_univariate_hessian(operator::_UnivariateOperator, x::T) where {T}
    ret = operator.f′′(x)
    check_return_type(T, ret)
    return ret::T
end

function eval_univariate_function_and_gradient(
    operator::_UnivariateOperator,
    x::T,
) where {T}
    ret_f = eval_univariate_function(operator, x)
    ret_f′ = eval_univariate_gradient(operator, x)
    return ret_f, ret_f′
end

function eval_univariate_function_and_gradient(
    registry::OperatorRegistry,
    id::Integer,
    x::T,
) where {T}
    if id <= registry.univariate_user_operator_start
        return _eval_univariate(id, x)::Tuple{T,T}
    end
    offset = id - registry.univariate_user_operator_start
    operator = registry.registered_univariate_operators[offset]
    return eval_univariate_function_and_gradient(operator, x)
end

function eval_multivariate_gradient(
    registry::OperatorRegistry,
    op::Symbol,
    g::AbstractVector{T},
    x::AbstractVector{T},
) where {T}
    @assert length(g) == length(x)
    if op == :+
        fill!(g, one(T))
    elseif op == :-
        g[1] = one(T)
        g[2] = -one(T)
    elseif op == :*
        # Special case performance optimizations for common cases.
        if length(x) == 1
            g[1] = one(T)
        elseif length(x) == 2
            g[1] = x[2]
            g[2] = x[1]
        else
            total = prod(x)
            if iszero(total)
                for i in eachindex(x)
                    g[i] = prod(x[j] for j in eachindex(x) if i != j)
                end
            else
                for i in eachindex(x)
                    g[i] = total / x[i]
                end
            end
        end
    elseif op == :^
        @assert length(x) == 2
        if x[2] == one(T)
            g[1] = one(T)
        elseif x[2] == T(2)
            g[1] = T(2) * x[1]
        else
            g[1] = x[2] * _nan_pow(x[1], x[2] - one(T))
        end
        if x[1] > zero(T)
            g[2] = _nan_pow(x[1], x[2]) * log(x[1])
        else
            g[2] = T(NaN)
        end
    elseif op == :/
        @assert length(x) == 2
        g[1] = one(T) / x[2]
        g[2] = -x[1] / x[2]^2
    elseif op == :ifelse
        @assert length(x) == 3
        g[1] = zero(T)  # It doesn't matter what this is.
        g[2] = x[1] == one(T)
        g[3] = x[1] == zero(T)
    elseif op == :atan
        @assert length(x) == 2
        base = x[1]^2 + x[2]^2
        g[1] = x[2] / base
        g[2] = -x[1] / base
    elseif op == :min
        fill!(g, zero(T))
        _, i = findmin(x)
        g[i] = one(T)
    elseif op == :max
        fill!(g, zero(T))
        _, i = findmax(x)
        g[i] = one(T)
    else
        id = registry.multivariate_operator_to_id[op]
        offset = id - registry.multivariate_user_operator_start
        operator = registry.registered_multivariate_operators[offset]
        @assert length(x) == operator.N
        operator.∇f(g, x)
    end
    return
end

function eval_comparison_function(
    ::OperatorRegistry,
    op::Symbol,
    lhs::T,
    rhs::T,
)::Bool where {T}
    if op == :<=
        return lhs <= rhs
    elseif op == :>=
        return lhs >= rhs
    elseif op == :(==)
        return lhs == rhs
    elseif op == :<
        return lhs < rhs
    else
        @assert op == :>
        return lhs > rhs
    end
end
