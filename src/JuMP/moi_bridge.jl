# Conversion from JuMP array types to MOI ArrayNonlinearFunction,
# to Julia Expr for ArrayDiff parsing, and NLPBlock setup helpers.

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

# ── Scalar expression from array operations ──────────────────────────────────

"""
    ArrayScalarExpr

A scalar-valued expression that operates on array subexpressions (e.g.,
`dot(A, B)`, `sum(A)`, `norm(A)`). This is the result type of scalar
reductions on `GenericArrayExpr`.
"""
struct ArrayScalarExpr
    head::Symbol
    args::Vector{Any}
end

function to_expr(x::ArrayScalarExpr)
    return Expr(:call, x.head, Any[to_expr(a) for a in x.args]...)
end

"""
    ArrayDiff.dot(x, y)

Compute the dot product (sum of elementwise products) of two array expressions.
Returns an `ArrayScalarExpr` (scalar).
"""
function dot(x, y)
    return ArrayScalarExpr(:dot, Any[x, y])
end

"""
    ArrayDiff.sumsq(x)

Compute the sum of squares of an array expression. Equivalent to `dot(x, x)`.
"""
function sumsq(x)
    return dot(x, x)
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

# ── NLPBlock setup helpers ───────────────────────────────────────────────────

"""
    set_nlp_objective!(jmodel::JuMP.Model, sense, objective)

Build an `ArrayDiff.Model` from the given `objective` expression (which may be
an `ArrayScalarExpr`, `GenericArrayExpr`, `ArrayNonlinearFunction`, or plain
`Expr`), create an `ArrayDiff.Evaluator` with first-order AD, and set the
resulting `MOI.NLPBlockData` on the JuMP model's backend.

## Example

```julia
model = Model(NLopt.Optimizer)
@variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
Y = W * X
diff = Y - target
ArrayDiff.set_nlp_objective!(model, MOI.MIN_SENSE, ArrayDiff.sumsq(diff))
optimize!(model)
```
"""
function set_nlp_objective!(
    jmodel::JuMP.Model,
    sense::MOI.OptimizationSense,
    objective,
)
    # Collect ordered variables
    vars = JuMP.all_variables(jmodel)
    ordered_variables = [JuMP.index(v) for v in vars]

    # Build ArrayDiff Model
    ad_model = Model()
    obj_expr = to_expr(objective)
    set_objective(ad_model, obj_expr)

    # Create evaluator (first-order AD)
    evaluator = Evaluator(ad_model, Mode(), ordered_variables)
    nlp_data = MOI.NLPBlockData(evaluator)

    # Set on the JuMP backend
    backend = JuMP.backend(jmodel)
    MOI.set(backend, MOI.NLPBlock(), nlp_data)
    MOI.set(backend, MOI.ObjectiveSense(), sense)
    return
end
