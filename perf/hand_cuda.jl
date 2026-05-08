module HandCuda

using BenchmarkTools

# Hand-written CUDA reverse-mode gradient of the 2-layer MLP loss
#
#   L = sum((W2 * tanh(W1 * X) - y) .^ 2)
#
# `W1 ∈ R^{h×d}` is the only differentiated input; `W2 ∈ R^{m×h}`,
# `X ∈ R^{d×n}`, `y ∈ R^{m×n}` are constants. The `/n` scaling is dropped to
# match the `arraydiff.jl` path, which can't currently express it cleanly
# because of a latent bug in ArrayDiff's `:/` operator on non-scalar tape
# offsets — a constant scaling factor changes nothing about the gradient
# pathway being measured.
#
# Buffers are preallocated in `Buffers{T}` and reused across calls (Hand-CUDA
# is meant as the lower-bound baseline for ArrayDiff, which preallocates its
# tape once at `MOI.initialize`). Five device buffers cover everything: `y_1`
# also holds `W1*X` (overwritten in place by `tanh`); `J_2` also holds
# `W2*y_1` (overwritten by `2 .* (J_2 .- y)`); `W2T_J2` is scaled in place by
# `J_1`.

using Random
using LinearAlgebra
using CUDA

# Output dimension `m`. Fixed at 2 to match `arraydiff.jl`, which can't parse
# a 1-row `:vcat` (the forward unpacks two children).
const OUT_DIM = 2

struct Buffers{M<:AbstractMatrix}
    y_1::M      # h × n   = tanh.(W1 * X)
    J_1::M      # h × n   = 1 .- y_1.^2
    J_2::M      # m × n   = 2 .* (W2*y_1 .- y)
    W2T_J2::M   # h × n   = W2' * J_2, then ⊙= J_1
    grad::M     # h × d   = W2T_J2 * X'
end

function Buffers{M}(h::Int, d::Int, n::Int) where {M}
    return Buffers{M}(
        M(undef, h, n),
        M(undef, h, n),
        M(undef, OUT_DIM, n),
        M(undef, h, n),
        M(undef, h, d),
    )
end

function gradient!(buf::Buffers, W1, W2, X, y)
    LinearAlgebra.mul!(buf.y_1, W1, X)            # y_1 = W1 * X
    buf.y_1 .= tanh.(buf.y_1)                     # y_1 = tanh.(y_1)
    buf.J_1 .= 1 .- buf.y_1 .^ 2
    LinearAlgebra.mul!(buf.J_2, W2, buf.y_1)
    buf.J_2 .= 2 .* (buf.J_2 .- y)
    LinearAlgebra.mul!(buf.W2T_J2, W2', buf.J_2)
    buf.W2T_J2 .= buf.J_1 .* buf.W2T_J2
    LinearAlgebra.mul!(buf.grad, buf.W2T_J2, X')
    return buf.grad
end

# Allocating reverse pass — same arithmetic, but every intermediate is a
# fresh `CuArray`. Useful as a "what does naive Julia broadcast cost?"
# baseline; expect ~6 transient allocations per call (one per dotted result).
function gradient_alloc(W1, W2, X, y)
    y_1 = tanh.(W1 * X)
    J_1 = 1 .- y_1 .^ 2
    J_2 = 2 .* (W2 * y_1 .- y)
    return (J_1 .* (W2' * J_2)) * X'
end

"""
    neural(T, h, d, n; prealloc::Bool = true) -> Matrix{T}

Compute `∂L/∂W1` of the 2-layer MLP loss with hand-written CUDA. Returns the
gradient as a host `Matrix{T}` of size `(h, d)`.

`prealloc = true` (default) reuses a single `Buffers{T}` across the forward
+ reverse pass — the apples-to-apples baseline against ArrayDiff (which
preallocates its tape once at `MOI.initialize`). `prealloc = false` runs the
same arithmetic with allocating dotted broadcasts, so each call mints fresh
`CuArray`s for every intermediate.
"""
function neural(
    ::Type{T},
    h::Int,
    d::Int,
    n::Int;
    prealloc::Bool = true,
    gpu::Bool = false,
) where {T<:Real}
    Random.seed!(0)
    W1 = randn(T, h, d)
    W2 = randn(T, OUT_DIM, h)
    X = randn(T, d, n)
    y = randn(T, OUT_DIM, n)
    if gpu
        W1g, W2g, Xg, yg = CuArray(W1), CuArray(W2), CuArray(X), CuArray(y)
        CUDA.synchronize()
    else
        W1g, W2g, Xg, yg = W1, W2, X, y
    end
    return @benchmark begin
        if $prealloc
            gradient!(Buffers{$(typeof(W1g))}($h, $d, $n), $W1g, $W2g, $Xg, $yg)
        else
            gradient_alloc($W1g, $W2g, $Xg, $yg)
        end
        if $gpu
            CUDA.synchronize()
        end
    end
end

end # module
