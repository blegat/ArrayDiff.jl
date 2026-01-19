# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    struct ColoringResult
        result::SMC.TreeSetColoringResult
        local_indices::Vector{Int}  # map from local to global indices
    end

Wrapper around TreeSetColoringResult that also stores local_indices mapping.
"""
struct ColoringResult{R<:SMC.AbstractColoringResult}
    result::R
    local_indices::Vector{Int}  # map from local to global indices
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

Returns `(I, J, result)` where `I` and `J` are the row and column indices
of the Hessian structure, and `result` is a `TreeSetColoringResult` from
SparseMatrixColorings.
"""
function _hessian_color_preprocess(
    edgelist,
    num_total_var,
    algo::SMC.GreedyColoringAlgorithm,
    seen_idx = MOI.Nonlinear.Coloring.IndexedSet(0),
)
    resize!(seen_idx, num_total_var)
    I, J = Int[], Int[]
    for (i, j) in edgelist
        push!(seen_idx, i)
        push!(seen_idx, j)
        push!(I, i)
        push!(J, j)
        if i != j
            push!(I, j)
            push!(J, i)
        end
    end
    local_indices = sort!(collect(seen_idx))
    empty!(seen_idx)

    # Handle empty case (no edges in Hessian)
    if isempty(local_indices)
        # Return empty structure - no variables to color
        # We still need to return a valid ColoringResult, but with empty local_indices
        # The I and J vectors are already empty, which is correct
        # For the result, we'll create a minimal valid structure with a diagonal element
        # Note: This case should rarely occur in practice
        S = SMC.SparsityPatternCSC(SparseArrays.spdiagm(0 => [true]))
        problem = SMC.ColoringProblem(;
            structure = :symmetric,
            partition = :column,
        )
        tree_result = SMC.coloring(S, problem, algo)
        result = ColoringResult(tree_result, Int[])
        return ones(length(local_indices)), I, J, result
    end

    global_to_local_idx = seen_idx.nzidx # steal for storage
    for k in eachindex(local_indices)
        global_to_local_idx[local_indices[k]] = k
    end
    # only do the coloring on the local indices
    for k in eachindex(I)
        I[k] = global_to_local_idx[I[k]]
        J[k] = global_to_local_idx[J[k]]
    end

    # Create sparsity pattern matrix
    n = length(local_indices)
    S = SMC.SparsityPatternCSC(
        SparseArrays.sparse(I, J, trues(length(I)), n, n, &)
    )

    # Perform coloring using SMC
    problem = SMC.ColoringProblem(;
        structure = :symmetric,
        partition = :column,
    )
    tree_result = SMC.coloring(S, problem, algo)

    # Wrap result with local_indices
    result = ColoringResult(tree_result, local_indices)

    # SparseMatrixColorings assumes that `I` and `J` are CSC-ordered
    B = SMC.compress(S, tree_result)
    C = SMC.decompress(B, tree_result)
    I_sorted, J_sorted = SparseArrays.findnz(C)

    return C.colptr, I_sorted, J_sorted, result
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
`stored_values` is a temporary vector.
"""
function _recover_from_matmat!(
    colptr::AbstractVector,
    V::AbstractVector{T},
    R::AbstractMatrix{T},
    result::ColoringResult,
    stored_values::AbstractVector{T},
) where {T}
    tree_result = result.result
    N = length(result.local_indices)
    S = _SparseMatrixValuesCSC(
        N,
        N,
        colptr,
        V,
    )
    SMC.decompress!(S, R, tree_result)
    return
end
