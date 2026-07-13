# Device-generic structured constant matrices for vectorized AC-OPF.
#
# Both types implement only what ArrayDiff's matmul nodes need —
# `LinearAlgebra.mul!(y, A, x)` and `LinearAlgebra.mul!(y, transpose(A), w)` —
# using nothing but broadcasts over (views of) their storage vectors, plus
# `cumsum!`. That makes the exact same code run on `Vector`, `JLArray` (the
# GPUArrays reference backend used to validate GPU semantics without a GPU),
# and `CuArray`, without any CUDA dependency here.
#
# * `GatherMatrix`: a 0/1 matrix with exactly one 1 per row. `A * x` is a pure
#   gather (`y .= x[idx]`, one fused kernel) and `A' * w` is a segmented sum
#   implemented with one gather, one `cumsum!` and one subtraction — no
#   atomics, no CSR machinery. This encodes branch↔bus / gen↔bus incidence.
# * `ELLMatrix`: fixed-width (padded) sparse rows. `A * x` is `K` fused
#   gather-multiply-accumulate broadcasts where `K` is the max number of
#   nonzeros per row (≈ node degree in a power network, small and uniform).
#   The transpose is stored explicitly as another `ELLMatrix`. This encodes
#   the bus admittance matrix and beats `SparseMatrixCSC`-style SpMV on GPU
#   for such uniformly-short rows because it needs no row-pointer indirection
#   and its memory access is coalesced.

import LinearAlgebra
import SparseArrays

# ── GatherMatrix ─────────────────────────────────────────────────────────────

"""
    GatherMatrix(idx::Vector{Int}, ncol::Int)

The `length(idx) × ncol` matrix `A` with `A[i, idx[i]] = 1` and zeros
elsewhere, so that `A * x = x[idx]` (a pure gather) and `A' * w` scatter-adds
`w` into the selected columns.

The transpose product is computed *without atomics and without a scan*:
column `j` collects the rows `{i : idx[i] == j}`; those row lists are padded
to the maximum count `D` (the maximum node degree, small in power networks)
into an `ncol × D` index matrix `tcols` pointing into a `length(idx) + 1`
buffer whose last slot is a permanent zero (the padding target). `A' * w` is
then `D` fused gather-add broadcasts.
"""
struct GatherMatrix{
    Vi<:AbstractVector{Int},
    Mi<:AbstractMatrix{Int},
    Vf<:AbstractVector{Float64},
} <: AbstractMatrix{Float64}
    idx::Vi
    ncol::Int
    tcols::Mi # ncol × D indices into `buf` (m + 1 = zero pad)
    buf::Vf   # length(idx) + 1 workspace with buf[end] == 0 kept invariant
end

function GatherMatrix(idx::AbstractVector{<:Integer}, ncol::Integer)
    m = length(idx)
    lists = [Int[] for _ in 1:ncol]
    for (i, j) in enumerate(idx)
        push!(lists[j], i)
    end
    D = maximum(length, lists; init = 0)
    tcols = fill(m + 1, ncol, D) # pad → permanent zero slot
    for j in 1:ncol, (k, i) in enumerate(lists[j])
        tcols[j, k] = i
    end
    return GatherMatrix(collect(Int, idx), Int(ncol), tcols, zeros(m + 1))
end

Base.size(A::GatherMatrix) = (length(A.idx), A.ncol)

# Only for `Matrix(A)` in reference computations / display; evaluation never
# uses scalar `getindex`.
Base.getindex(A::GatherMatrix, i::Int, j::Int) = Float64(A.idx[i] == j)

function LinearAlgebra.mul!(
    y::AbstractVector,
    A::GatherMatrix,
    x::AbstractVector,
)
    y .= view(x, A.idx)
    return y
end

function LinearAlgebra.mul!(
    y::AbstractVector,
    At::LinearAlgebra.Transpose{Float64,<:GatherMatrix},
    w::AbstractVector,
)
    A = LinearAlgebra.parent(At)
    m = length(A.idx)
    D = size(A.tcols, 2)
    if D == 0
        fill!(y, 0.0)
        return y
    end
    view(A.buf, 1:m) .= w # buf[m + 1] stays 0 (padding target)
    y .= view(A.buf, view(A.tcols, :, 1))
    for k in 2:D
        y .+= view(A.buf, view(A.tcols, :, k))
    end
    return y
end

# ── ELLMatrix ────────────────────────────────────────────────────────────────

"""
    ELLMatrix(S::SparseMatrixCSC)

ELLPACK-format copy of `S`: row `i`'s nonzeros are `vals[i, k]` at columns
`cols[i, k]` for `k = 1:K`, padded with zero values pointing at column 1.
The transpose is stored eagerly in `at` so that reverse-mode `A' * w`
products use the same kernel.
"""
struct ELLMatrix{
    Mi<:AbstractMatrix{Int},
    Mf<:AbstractMatrix{Float64},
    TT,
} <: AbstractMatrix{Float64}
    m::Int
    n::Int
    cols::Mi
    vals::Mf
    at::TT # ELLMatrix of the transpose, or `nothing`
end

function ELLMatrix(S::SparseArrays.SparseMatrixCSC; with_transpose::Bool = true)
    m, n = size(S)
    I, J, V = SparseArrays.findnz(S)
    counts = zeros(Int, m)
    for i in I
        counts[i] += 1
    end
    K = maximum(counts; init = 0)
    cols = ones(Int, m, K)
    vals = zeros(m, K)
    fill!(counts, 0)
    for (i, j, v) in zip(I, J, V)
        counts[i] += 1
        cols[i, counts[i]] = j
        vals[i, counts[i]] = v
    end
    at = if with_transpose
        ELLMatrix(
            SparseArrays.sparse(LinearAlgebra.transpose(S));
            with_transpose = false,
        )
    else
        nothing
    end
    return ELLMatrix(m, n, cols, vals, at)
end

Base.size(A::ELLMatrix) = (A.m, A.n)

function Base.getindex(A::ELLMatrix, i::Int, j::Int)
    acc = 0.0
    for k in 1:size(A.cols, 2)
        if A.cols[i, k] == j
            acc += A.vals[i, k]
        end
    end
    return acc
end

function LinearAlgebra.mul!(y::AbstractVector, A::ELLMatrix, x::AbstractVector)
    K = size(A.cols, 2)
    if K == 0
        fill!(y, 0.0)
        return y
    end
    y .= view(A.vals, :, 1) .* view(x, view(A.cols, :, 1))
    for k in 2:K
        y .+= view(A.vals, :, k) .* view(x, view(A.cols, :, k))
    end
    return y
end

function LinearAlgebra.mul!(
    y::AbstractVector,
    At::LinearAlgebra.Transpose{Float64,<:ELLMatrix},
    w::AbstractVector,
)
    A = LinearAlgebra.parent(At)
    A.at === nothing && error("ELLMatrix built without transpose")
    return LinearAlgebra.mul!(y, A.at, w)
end

# ── Device transfer ──────────────────────────────────────────────────────────

"""
    map_storage(f, A)

Rebuild `A` with every storage array passed through `f` (for example
`f = CUDA.CuArray` or `f = JLArrays.JLArray`). `f = identity` is a no-op.
"""
map_storage(f, A::AbstractArray) = f(A)

function map_storage(f, A::GatherMatrix)
    return GatherMatrix(f(A.idx), A.ncol, f(A.tcols), f(A.buf))
end

function map_storage(f, A::ELLMatrix)
    at = A.at === nothing ? nothing : map_storage(f, A.at)
    return ELLMatrix(A.m, A.n, f(A.cols), f(A.vals), at)
end

function map_storage(f, At::LinearAlgebra.Transpose)
    return LinearAlgebra.transpose(map_storage(f, LinearAlgebra.parent(At)))
end
