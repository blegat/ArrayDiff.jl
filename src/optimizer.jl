# ArrayDiff.Optimizer — MOI optimizer wrapper that handles ArrayNonlinearFunction
# objectives by converting them to NLPBlock via ArrayDiff's evaluator.

"""
    ArrayDiff.Optimizer(inner::MOI.AbstractOptimizer)

Wrap an MOI optimizer to add support for `ArrayNonlinearFunction` objectives.
When an `ArrayNonlinearFunction` objective is set, it is parsed into an
`ArrayDiff.Model` and converted to `MOI.NLPBlock` on the inner optimizer.

## Usage with JuMP

```julia
model = Model(() -> ArrayDiff.Optimizer(NLopt.Optimizer()))
```
"""
mutable struct Optimizer{O<:MOI.AbstractOptimizer} <: MOI.AbstractOptimizer
    inner::O
    ad_model::Union{Nothing,Model}
end

function Optimizer(inner::O) where {O<:MOI.AbstractOptimizer}
    return Optimizer{O}(inner, nothing)
end

# ── Objective: intercept ArrayNonlinearFunction, forward everything else ─────

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{ArrayNonlinearFunction{N}},
) where {N}
    return true
end

function MOI.set(
    opt::Optimizer,
    ::MOI.ObjectiveFunction{ArrayNonlinearFunction{N}},
    f::ArrayNonlinearFunction{N},
) where {N}
    ad_model = Model()
    set_objective(ad_model, f)
    opt.ad_model = ad_model
    return
end

function MOI.optimize!(opt::Optimizer)
    if opt.ad_model !== nothing
        vars = MOI.get(opt.inner, MOI.ListOfVariableIndices())
        evaluator = Evaluator(opt.ad_model, Mode(), vars)
        nlp_data = MOI.NLPBlockData(evaluator)
        MOI.set(opt.inner, MOI.NLPBlock(), nlp_data)
        MOI.set(opt.inner, MOI.ObjectiveSense(), MOI.get(opt, MOI.ObjectiveSense()))
    end
    return MOI.optimize!(opt.inner)
end

# ── Forward all other MOI operations to inner ────────────────────────────────

MOI.add_variable(opt::Optimizer) = MOI.add_variable(opt.inner)
MOI.add_variables(opt::Optimizer, n) = MOI.add_variables(opt.inner, n)

function MOI.add_constraint(opt::Optimizer, f::F, s::S) where {F,S}
    return MOI.add_constraint(opt.inner, f, s)
end

function MOI.supports_constraint(
    opt::Optimizer,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports_constraint(opt.inner, F, S)
end

function MOI.supports(
    opt::Optimizer,
    attr::MOI.AbstractOptimizerAttribute,
)
    return MOI.supports(opt.inner, attr)
end

function MOI.supports(
    opt::Optimizer,
    attr::MOI.AbstractModelAttribute,
)
    return MOI.supports(opt.inner, attr)
end

function MOI.set(opt::Optimizer, attr::MOI.AbstractOptimizerAttribute, value)
    return MOI.set(opt.inner, attr, value)
end

function MOI.set(opt::Optimizer, attr::MOI.AbstractModelAttribute, value)
    return MOI.set(opt.inner, attr, value)
end

function MOI.set(
    opt::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
    value,
)
    return MOI.set(opt.inner, attr, vi, value)
end

function MOI.get(opt::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    return MOI.get(opt.inner, attr)
end

function MOI.get(opt::Optimizer, attr::MOI.AbstractModelAttribute)
    return MOI.get(opt.inner, attr)
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
)
    return MOI.get(opt.inner, attr, vi)
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::Vector{MOI.VariableIndex},
)
    return MOI.get(opt.inner, attr, vi)
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
)
    return MOI.get(opt.inner, attr, ci)
end

function MOI.is_empty(opt::Optimizer)
    return MOI.is_empty(opt.inner) && opt.ad_model === nothing
end

function MOI.empty!(opt::Optimizer)
    MOI.empty!(opt.inner)
    opt.ad_model = nothing
    return
end

function MOI.get(opt::Optimizer, ::MOI.ListOfVariableIndices)
    return MOI.get(opt.inner, MOI.ListOfVariableIndices())
end

function MOI.get(opt::Optimizer, ::MOI.NumberOfVariables)
    return MOI.get(opt.inner, MOI.NumberOfVariables())
end

function MOI.supports(
    opt::Optimizer,
    attr::MOI.ObjectiveFunction{F},
) where {F<:MOI.AbstractScalarFunction}
    return MOI.supports(opt.inner, attr)
end

function MOI.set(
    opt::Optimizer,
    attr::MOI.ObjectiveFunction{F},
    f::F,
) where {F<:MOI.AbstractScalarFunction}
    return MOI.set(opt.inner, attr, f)
end
