# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    struct ColoringResult
        result::SMC.AbstractColoringResult
        local_indices::Vector{Int}  # map from local to global indices
        full_colptr::Vector{Int}    # colptr of the full symmetric matrix used for coloring
        lower_pos::Vector{Int}      # positions of lower-triangular entries in the full nzval
        full_buffer::Vector{Float64} # pre-allocated buffer of size nnz(full symmetric matrix)
    end

Wrapper around AbstractColoringResult that also stores auxiliary data needed
for Hessian recovery from a full symmetric matrix decompression.
"""
struct ColoringResult{R<:SMC.AbstractColoringResult}
    result::R
    local_indices::Vector{Int}   # map from local to global indices
    full_colptr::Vector{Int}     # colptr of full symmetric matrix used for coloring
    lower_pos::Vector{Int}       # positions of lower-triangular entries in full nzval
    full_buffer::Vector{Float64} # scratch buffer of length nnz(full symmetric matrix)
end

"""
    _hessian_color_preprocess(
        edgelist,
        num_total_var,
        algo::SMC.GreedyColoringAlgorithm,
        seen_idx = MOI.Nonlinear.Coloring.IndexedSet(0),
    )

`edgelist` contains the nonzeros in the Hessian, *including* nonzeros on the
diagonal.

Returns `(colptr, I, J, result)` where `colptr`, `I` and `J` define the lower
triangular CSC sparsity structure of the Hessian (in global variable indices),
and `result` is a `ColoringResult` wrapping an SMC coloring result.
"""
function _hessian_color_preprocess(
    edgelist,
    num_total_var,
    algo::SMC.GreedyColoringAlgorithm,
    seen_idx = MOI.Nonlinear.Coloring.IndexedSet(0),
)
    resize!(seen_idx, num_total_var)
    # Collect off-diagonal lower-triangular entries (local coords, filled later)
    I_off, J_off = Int[], Int[]
    for (ei, ej) in edgelist
        push!(seen_idx, ei)
        push!(seen_idx, ej)
        if ei != ej
            # Store in lower triangular format: row > col
            push!(I_off, max(ei, ej))
            push!(J_off, min(ei, ej))
        end
    end
    local_indices = sort!(collect(seen_idx))
    empty!(seen_idx)

    global_to_local_idx = seen_idx.nzidx # steal for storage
    for k in eachindex(local_indices)
        global_to_local_idx[local_indices[k]] = k
    end
    # Map off-diagonal entries to local indices
    for k in eachindex(I_off)
        I_off[k] = global_to_local_idx[I_off[k]]
        J_off[k] = global_to_local_idx[J_off[k]]
    end

    n = length(local_indices)

    # Build full symmetric matrix: both (i,j) and (j,i) for off-diagonal, plus diagonal
    I_full, J_full = Int[], Int[]
    for k in eachindex(I_off)
        push!(I_full, I_off[k]);
        push!(J_full, J_off[k])  # lower
        push!(I_full, J_off[k]);
        push!(J_full, I_off[k])  # upper (transpose)
    end
    for k in 1:n
        push!(I_full, k);
        push!(J_full, k)  # diagonal
    end
    mat_sym =
        SparseArrays.sparse(I_full, J_full, trues(length(I_full)), n, n, |)

    # Perform coloring on full symmetric matrix
    S = SMC.SparsityPatternCSC(mat_sym)
    problem = SMC.ColoringProblem(; structure = :symmetric, partition = :column)
    tree_result = SMC.coloring(S, problem, algo)

    # Find positions of lower-triangular entries within the full CSC nzval array.
    # findnz on a CSC matrix returns elements in CSC (column-major) order,
    # matching the nzval layout exactly.
    I_nz, J_nz, _ = SparseArrays.findnz(mat_sym)
    lower_pos = findall(k -> I_nz[k] >= J_nz[k], eachindex(I_nz))

    # Lower-triangular CSC-ordered local indices
    I_low_csc = I_nz[lower_pos]
    J_low_csc = J_nz[lower_pos]

    # Map back to global indices for the returned hess_I / hess_J
    I_global = [local_indices[i] for i in I_low_csc]
    J_global = [local_indices[j] for j in J_low_csc]

    # Build lower-triangular sparse matrix to obtain its colptr
    mat_low = SparseArrays.sparse(
        I_low_csc,
        J_low_csc,
        trues(length(I_low_csc)),
        n,
        n,
        |,
    )

    full_buffer = Vector{Float64}(undef, SparseArrays.nnz(mat_sym))

    result = ColoringResult(
        tree_result,
        local_indices,
        copy(mat_sym.colptr),
        lower_pos,
        full_buffer,
    )

    return copy(mat_low.colptr), I_global, J_global, result
end

"""
    _seed_matrix(result::ColoringResult)

Allocate a seed matrix for the coloring result.
"""
function _seed_matrix(result::ColoringResult)
    n = length(result.local_indices)
    ncolors = SMC.ncolors(result.result)
    return Matrix{Float64}(undef, n, ncolors)
end

"""
    _prepare_seed_matrix!(R, result::ColoringResult)

Prepare the seed matrix R for Hessian computation.
"""
function _prepare_seed_matrix!(R, result::ColoringResult)
    color = SMC.column_colors(result.result)
    N = length(result.local_indices)
    @assert N == size(R, 1)
    @assert size(R, 2) == SMC.ncolors(result.result)
    fill!(R, 0.0)
    for i in 1:N
        if color[i] > 0
            R[i, color[i]] = 1
        end
    end
    return
end

"""
    _recover_from_matmat!(
        colptr::AbstractVector,
        V::AbstractVector{T},
        R::AbstractMatrix{T},
        result::ColoringResult,
        stored_values::AbstractVector{T},
    ) where {T}

Recover the Hessian values from the Hessian-matrix product H*R_seed.
R is the result of H*R_seed where R_seed is the seed matrix.
`stored_values` is a temporary vector (unused, kept for API compatibility).
"""
function _recover_from_matmat!(
    colptr::AbstractVector,
    V::AbstractVector{T},
    R::AbstractMatrix{T},
    result::ColoringResult,
    stored_values::AbstractVector{T},
) where {T}
    # Decompress into the full symmetric buffer, then extract lower-triangular values.
    SMC.decompress_csc!(
        result.full_buffer,
        result.full_colptr,
        R,
        result.result,
        :F,
    )
    for k in eachindex(V)
        V[k] = result.full_buffer[result.lower_pos[k]]
    end
    return
end
