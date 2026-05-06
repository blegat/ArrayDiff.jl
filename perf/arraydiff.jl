module ArrayDiffNeural

# ArrayDiff reverse-mode AD of the 2-layer MLP loss
#
#   L = sum((W2 * tanh(W1 * X) - y) .^ 2)
#
# `W1` is the only `@variable`; `W2`, `X`, `y` are baked in as constant
# matrix blocks. With `gpu=true`, the AD tape, input vector, and gradient
# buffer all live on the device (`Mode{CuVector{T}}`); with `gpu=false`,
# everything stays on the host (`Mode{Vector{T}}`).
#
# `/n` is dropped from the loss to match `hand_cuda.jl` — ArrayDiff's `:/`
# operator currently has a latent bug on non-scalar tape offsets, and a
# constant scaling factor doesn't change the gradient pathway.

using Random
using BenchmarkTools
using CUDA
using JuMP
using ArrayDiff
import MathOptInterface as MOI

# Output dimension `m`. Fixed at 2: a 1-row `W2` would parse as a `:vcat`
# with a single child, which the current ArrayDiff parser unpacks via
# `idx1, idx2 = children_indices` and so requires at least two rows.
const OUT_DIM = 2

function _build(::Type{T}, h::Int, d::Int, n::Int, gpu::Bool) where {T<:Real}
    Random.seed!(0)
    W1 = randn(T, h, d)
    W2 = randn(T, OUT_DIM, h)
    X = randn(T, d, n)
    y = randn(T, OUT_DIM, n)

    # JuMP works in Float64 internally; ArrayDiff converts the parsed
    # `Expression{Float64}` constants into the `Mode`'s eltype when filling
    # its tape. Promote constants to Float64 explicitly so the conversion
    # is one-way and obvious.
    W2_64 = Float64.(W2)
    X_64 = Float64.(X)
    y_64 = Float64.(y)

    model = JuMP.Model()
    @variable(model, W1v[1:h, 1:d], container = ArrayDiff.ArrayOfVariables)
    Y = W2_64 * tanh.(W1v * X_64)
    diff = Y .- y_64
    loss = sum(diff .^ 2)

    mode = gpu ?
        ArrayDiff.Mode{CUDA.CuVector{T}}() :
        ArrayDiff.Mode{Vector{T}}()
    ad = ArrayDiff.model(mode)
    MOI.Nonlinear.set_objective(ad, JuMP.moi_function(loss))
    ev = MOI.Nonlinear.Evaluator(
        ad,
        mode,
        JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.initialize(ev, [:Grad])
    return (W1 = W1, evaluator = ev)
end

"""
    neural(T, h, d, n; gpu::Bool = false) -> BenchmarkTools.Trial

Benchmark `∂L/∂W1` of the 2-layer MLP loss using ArrayDiff. Model parsing
and `MOI.initialize` (the expensive per-shape build) happen once before the
trial; the timed block is just `MOI.eval_objective_gradient` plus a
`CUDA.synchronize` when `gpu=true`.
"""
function neural(
    ::Type{T},
    h::Int,
    d::Int,
    n::Int;
    gpu::Bool = false,
) where {T<:Real}
    state = _build(T, h, d, n, gpu)
    if gpu
        x = CUDA.CuVector{T}(vec(state.W1))
        g = CUDA.zeros(T, h * d)
    else
        x = vec(state.W1)
        g = zeros(T, h * d)
    end
    return @benchmark begin
        if $gpu
            # `@allowscalar` covers residual scalar leaves the BLOCK rewrite
            # can't fold; the hot path is bulk.
            CUDA.@allowscalar MOI.eval_objective_gradient(
                $state.evaluator,
                $g,
                $x,
            )
            CUDA.synchronize()
        else
            MOI.eval_objective_gradient($state.evaluator, $g, $x)
        end
    end
end

end # module
