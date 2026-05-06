module LuxNeural

# Lux + Mooncake reverse-mode AD of the 2-layer MLP loss
#
#   L = sum((W2 * tanh(W1 * X) - y) .^ 2)
#
# Same arithmetic as the other `perf/*.jl` paths, expressed as a Lux `Chain`
# of two bias-less `Dense` layers (`tanh` then `identity`) and differentiated
# with `Mooncake.value_and_gradient!!`. Goes through the same CUDA.jl /
# cuBLAS stack as `hand_cuda.jl` — what we're isolating is the AD/dispatch
# overhead that Lux + Mooncake add on top.

using Random
using BenchmarkTools
using CUDA
using Lux
import Mooncake

const OUT_DIM = 2

function _build(::Type{T}, h::Int, d::Int, n::Int, gpu::Bool) where {T<:Real}
    Random.seed!(0)
    W1 = randn(T, h, d)
    W2 = randn(T, OUT_DIM, h)
    X = randn(T, d, n)
    y = randn(T, OUT_DIM, n)
    if gpu
        W1, W2, X, y = CuArray(W1), CuArray(W2), CuArray(X), CuArray(y)
        CUDA.synchronize()
    end

    model = Lux.Chain(
        Lux.Dense(d => h, tanh; use_bias = false),
        Lux.Dense(h => OUT_DIM, identity; use_bias = false),
    )
    ps = (layer_1 = (weight = W1,), layer_2 = (weight = W2,))
    st = Lux.initialstates(Random.default_rng(), model)

    # Closure captures everything except `p`, which is what we differentiate.
    loss_fn = let model = model, st = st, X = X, y = y
        p -> begin
            y_hat, _ = model(X, p, st)
            return sum((y_hat .- y) .^ 2)
        end
    end
    rule = Mooncake.build_rrule(loss_fn, ps)
    return (rule = rule, loss_fn = loss_fn, ps = ps)
end

"""
    neural(T, h, d, n; gpu::Bool = false) -> BenchmarkTools.Trial

Benchmark `∂L/∂W1` of the 2-layer MLP loss using Lux + Mooncake.
`Mooncake.build_rrule` (the expensive per-shape compile) happens once before
the trial; the timed block is just `value_and_gradient!!` plus a
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
    return @benchmark begin
        Mooncake.value_and_gradient!!($state.rule, $state.loss_fn, $state.ps)
        if $gpu
            CUDA.synchronize()
        end
    end
end

end # module
