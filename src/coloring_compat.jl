# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    struct ColoringResult
        result::SparseMatrixColorings.TreeSetColoringResult
        local_indices::Vector{Int}  # map from local to global indices
    end

Wrapper around TreeSetColoringResult that also stores local_indices mapping.
"""
struct ColoringResult
    result::SparseMatrixColorings.TreeSetColoringResult
    local_indices::Vector{Int}  # map from local to global indices
end

"""
    _hessian_color_preprocess(
        edgelist,
        num_total_var,
        seen_idx = IndexedSet(0),
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
    seen_idx = IndexedSet(0),
)
    resize!(seen_idx, num_total_var)
    I, J = Int[], Int[]
    for (i, j) in edgelist
        push!(seen_idx, i)
        push!(seen_idx, j)
        push!(I, i)
        push!(J, j)
    end
    local_indices = sort!(collect(seen_idx))
    empty!(seen_idx)
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
    S = SparseArrays.spzeros(Bool, n, n)
    for k in eachindex(I)
        i, j = I[k], J[k]
        S[i, j] = true
        S[j, i] = true  # symmetric
    end
    
    # Perform coloring using SparseMatrixColorings
    problem = SparseMatrixColorings.ColoringProblem(; structure=:symmetric, partition=:column)
    algo = SparseMatrixColorings.GreedyColoringAlgorithm(; decompression=:substitution)
    tree_result = SparseMatrixColorings.coloring(S, problem, algo)
    
    # Convert back to global indices
    for k in eachindex(I)
        I[k] = local_indices[I[k]]
        J[k] = local_indices[J[k]]
    end
    
    # Wrap result with local_indices
    result = ColoringResult(tree_result, local_indices)
    return I, J, result
end

"""
    _seed_matrix(result::ColoringResult)

Allocate a seed matrix for the coloring result.
"""
function _seed_matrix(result::ColoringResult)
    n = length(result.local_indices)
    ncolors = SparseMatrixColorings.ncolors(result.result)
    return Matrix{Float64}(undef, n, ncolors)
end

"""
    _prepare_seed_matrix!(R, result::ColoringResult)

Prepare the seed matrix R for Hessian computation.
"""
function _prepare_seed_matrix!(R, result::ColoringResult)
    color = SparseMatrixColorings.column_colors(result.result)
    N = length(result.local_indices)
    @assert N == size(R, 1)
    @assert size(R, 2) == SparseMatrixColorings.ncolors(result.result)
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
    V::AbstractVector{T},
    R::AbstractMatrix{T},
    result::ColoringResult,
    stored_values::AbstractVector{T},
) where {T}
    tree_result = result.result
    color = SparseMatrixColorings.column_colors(tree_result)
    N = length(result.local_indices)
    S = tree_result.ag.S
    # Compute number of off-diagonal nonzeros from the length of V
    # V contains N diagonal elements + nnz_offdiag off-diagonal elements
    nnz_offdiag = length(V) - N
    @assert length(stored_values) >= N
    
    # Recover diagonal elements
    k = 0
    for i in 1:N
        k += 1
        if color[i] > 0
            V[k] = R[i, color[i]]
        else
            V[k] = zero(T)
        end
    end
    
    # Recover off-diagonal elements using tree structure
    (; reverse_bfs_orders, tree_edge_indices, nt) = tree_result
    fill!(stored_values, zero(T))
    
    for tree_idx in 1:nt
        first = tree_edge_indices[tree_idx]
        last = tree_edge_indices[tree_idx + 1] - 1
        
        # Reset stored_values for vertices in this tree
        for pos in first:last
            (vertex, _) = reverse_bfs_orders[pos]
            stored_values[vertex] = zero(T)
        end
        (_, root) = reverse_bfs_orders[last]
        stored_values[root] = zero(T)
        
        # Recover edge values
        for pos in first:last
            (i, j) = reverse_bfs_orders[pos]
            if color[j] > 0
                value = R[i, color[j]] - stored_values[i]
            else
                value = zero(T)
            end
            stored_values[j] += value
            k += 1
            V[k] = value
        end
    end
    
    @assert k == length(V)
    return
end
