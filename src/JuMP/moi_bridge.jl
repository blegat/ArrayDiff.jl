# Conversion from JuMP array types to MOI ArrayNonlinearFunction,
# to Julia Expr for ArrayDiff parsing, and NLPBlock setup via
# JuMP.set_objective_function override.

# ── moi_function: JuMP → MOI ─────────────────────────────────────────────────

function _to_moi_arg(x::ArrayOfVariables{T,N}) where {T,N}
    return ArrayOfVariableIndices{N}(x.offset, x.size)
end

function _to_moi_arg(x::GenericArrayExpr{V,N}) where {V,N}
    args = Any[_to_moi_arg(a) for a in x.args]
    return ArrayNonlinearFunction{N}(x.head, args, x.size, x.broadcasted)
end

function _to_moi_arg(x::Matrix{Float64})
    return x
end

function _to_moi_arg(x::Real)
    return Float64(x)
end

function JuMP.moi_function(x::GenericArrayExpr{V,N}) where {V,N}
    return _to_moi_arg(x)
end

# ── to_expr: convert to Julia Expr for ArrayDiff.parse_expression ────────────

"""
    to_expr(x)

Convert a JuMP array expression (or MOI function) to a Julia `Expr` that can
be fed into `ArrayDiff.parse_expression`.
"""
function to_expr end

function to_expr(x::ArrayOfVariables{T,2}) where {T}
    m, n = size(x)
    rows = Any[]
    for i in 1:m
        cols = Any[MOI.VariableIndex(x.offset + (j - 1) * m + i) for j in 1:n]
        push!(rows, Expr(:row, cols...))
    end
    return Expr(:vcat, rows...)
end

function to_expr(x::ArrayOfVariables{T,1}) where {T}
    m = size(x, 1)
    elems = Any[MOI.VariableIndex(x.offset + i) for i in 1:m]
    return Expr(:vect, elems...)
end

function to_expr(x::ArrayOfVariableIndices{2})
    m, n = x.size
    rows = Any[]
    for i in 1:m
        cols = Any[MOI.VariableIndex(x.offset + (j - 1) * m + i) for j in 1:n]
        push!(rows, Expr(:row, cols...))
    end
    return Expr(:vcat, rows...)
end

function to_expr(x::ArrayOfVariableIndices{1})
    m = x.size[1]
    elems = Any[MOI.VariableIndex(x.offset + i) for i in 1:m]
    return Expr(:vect, elems...)
end

function to_expr(x::Matrix{Float64})
    m, n = size(x)
    rows = Any[]
    for i in 1:m
        push!(rows, Expr(:row, x[i, :]...))
    end
    return Expr(:vcat, rows...)
end

function to_expr(x::Vector{Float64})
    return Expr(:vect, x...)
end

function to_expr(x::Real)
    return Float64(x)
end

function to_expr(x::MOI.VariableIndex)
    return x
end

function to_expr(x::GenericArrayExpr)
    if x.broadcasted && length(x.args) == 1
        # Univariate broadcasted: Expr(:., :tanh, Expr(:tuple, child))
        return Expr(:., x.head, Expr(:tuple, to_expr(x.args[1])))
    elseif x.broadcasted
        # Multivariate broadcasted: Expr(:call, Symbol(".*"), args...)
        dothead = Symbol("." * string(x.head))
        return Expr(:call, dothead, Any[to_expr(a) for a in x.args]...)
    else
        return Expr(:call, x.head, Any[to_expr(a) for a in x.args]...)
    end
end

function to_expr(x::ArrayNonlinearFunction)
    if x.broadcasted && length(x.args) == 1
        return Expr(:., x.head, Expr(:tuple, to_expr(x.args[1])))
    elseif x.broadcasted
        dothead = Symbol("." * string(x.head))
        return Expr(:call, dothead, Any[to_expr(a) for a in x.args]...)
    else
        return Expr(:call, x.head, Any[to_expr(a) for a in x.args]...)
    end
end

function to_expr(x::Expr)
    return x
end

# ── to_expr for JuMP scalar nonlinear expressions ────────────────────────────

function to_expr(x::JuMP.GenericNonlinearExpr)
    return Expr(:call, x.head, Any[to_expr(a) for a in x.args]...)
end

function to_expr(x::JuMP.GenericVariableRef)
    return JuMP.index(x)
end

# ── parse_expression for ArrayNonlinearFunction ──────────────────────────────

function parse_expression(
    data::Model,
    expr::Expression,
    x::ArrayNonlinearFunction,
    parent_index::Int,
)
    return parse_expression(data, expr, to_expr(x), parent_index)
end

function parse_expression(
    data::Model,
    expr::Expression,
    x::ArrayOfVariableIndices,
    parent_index::Int,
)
    return parse_expression(data, expr, to_expr(x), parent_index)
end

# ── Detect whether a JuMP expression contains array args ─────────────────────

_has_array_args(::Any) = false
_has_array_args(::AbstractJuMPArray) = true
_has_array_args(::ArrayNonlinearFunction) = true

function _has_array_args(x::JuMP.GenericNonlinearExpr)
    return any(_has_array_args, x.args)
end

# ── Override set_objective_function for array-valued nonlinear expressions ────

function _set_arraydiff_nlp_block!(
    jmodel::JuMP.GenericModel{T},
    func::JuMP.GenericNonlinearExpr{JuMP.GenericVariableRef{T}},
) where {T}
    vars = JuMP.all_variables(jmodel)
    ordered_variables = [JuMP.index(v) for v in vars]
    ad_model = Model()
    obj_expr = to_expr(func)
    set_objective(ad_model, obj_expr)
    evaluator = Evaluator(ad_model, Mode(), ordered_variables)
    nlp_data = MOI.NLPBlockData(evaluator)
    MOI.set(JuMP.backend(jmodel), MOI.NLPBlock(), nlp_data)
    return
end

function JuMP.set_objective_function(
    model::JuMP.GenericModel{T},
    func::JuMP.GenericNonlinearExpr{JuMP.GenericVariableRef{T}},
) where {T<:Real}
    if _has_array_args(func)
        return _set_arraydiff_nlp_block!(model, func)
    end
    # Fall back to standard JuMP: convert to MOI and set on backend.
    f = JuMP.moi_function(func)
    attr = MOI.ObjectiveFunction{typeof(f)}()
    MOI.set(JuMP.backend(model), attr, f)
    return
end
