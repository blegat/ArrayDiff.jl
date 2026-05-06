# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    _reverse_mode(d::NLPEvaluator, x)

Run reverse-mode automatic differentiation on `d` given the primal solution `x`.

This function updates many of the data-structures inside `d` in-place.

At a high level, reverse-mode AD has two phases:

In Phase I, we evaluate the problem in `d` at the primal solution `x`, and
stores the primal solution of each expression in the tree and the first-order
partial derivative information for each node with respect to its arguments.

Because the nodes in our data structure are topologically sorted, we can make a
single pass through the tree by iterating backwards through the vector of stored
nodes.

In Phase II, we propagate the partial derivative information back down the tree
to find the derivative of each function with respect to the input.

Because the nodes in our data structure are topologically sorted, we can make a
single pass through the tree by iterating forwards through the vector of stored
nodes.
"""
function _reverse_mode(d::NLPEvaluator, x)
    # Because the operators are checked with `Int` and not `Symbol`
    # if we get a model that didn't add our new operators but had user-defined
    # operators, we will think that these are one of our new operators
    @assert :vect in d.data.operators.multivariate_operators
    if d.last_x == x
        # Fail fast if the primal solution has not changed since last call.
        return
    end
    # Phase I
    for k in d.subexpression_order
        d.subexpression_forward_values[k] =
            _forward_eval(d.subexpressions[k], d, x)
    end
    if d.objective !== nothing
        _forward_eval(something(d.objective).expr, d, x)
    end
    for con in d.constraints
        _forward_eval(con.expr, d, x)
    end
    # Phase II
    for k in d.subexpression_order
        _reverse_eval(d.subexpressions[k])
    end
    if d.objective !== nothing
        _reverse_eval(something(d.objective).expr)
    end
    for con in d.constraints
        _reverse_eval(con.expr)
    end
    # If a JuMP model uses the legacy nonlinear interface, then JuMP constructs
    # a NLPEvaluator at the start of a call to `JuMP.optimize!` and it passes in
    # the list of variables in the JuMP model to `.ordered_variables`.
    #
    # During `MOI.initialize`, `.last_x` gets filled with `NaN` to match the
    # length of `ordered_variables`, that is, the number of variables in the
    # JuMP model.
    #
    # However, if the model includes a bridge that adds new decision variables
    # then the total number of variables in the optimizer (in `x`) will be
    # larger than the cache in `last_x`.
    #
    # It is safe to resize `last_x` because only the variables in
    # `ordered_variables` can appear in the NLPBlock.
    #
    # I don't think we need any other fixes because callers to things like
    # `eval_objective` can pass in a longer input `x` vector without fear
    # because the excess elements won't be used.
    if length(d.last_x) < length(x)
        resize!(d.last_x, length(x))
    end
    copyto!(d.last_x, x)
    return
end

"""
    _forward_eval(
        f::_SubexpressionStorage,
        d::NLPEvaluator,
        x::AbstractVector{T},
    ) where {T}

Forward-mode evaluation of an expression tree given in `f`.

 * This function assumes that the values of all sub-expressions have already
   been computed and are stored in `d.subexpression_forward_values`.
 * `f.partials_storage[k]` is the partial derivative of `nodes[k].parent` with
   respect to the value of node `k`. It's efficient to compute this at the same
   time as the value of the parent because we use it in reverse mode and in dual
   forward mode. Note that `partials_storage`` makes a subtle assumption that we
   have a tree instead of a general DAG. If we have a DAG, then need to
   associate storage with each edge of the DAG.
"""
function _forward_eval(
    f::_SubexpressionStorage,
    d::NLPEvaluator,
    x::AbstractVector{T},
)::T where {T}
    @assert length(f.forward_storage) >= length(f.nodes)
    @assert length(f.partials_storage) >= length(f.nodes)
    operators = d.data.operators
    # f.nodes is already in order such that parents always appear before
    # children, so a backwards pass through f.nodes is a forward pass through
    # the tree.
    children_arr = SparseArrays.rowvals(f.adj)
    fill!(f.partials_storage, zero(T))
    for k in length(f.nodes):-1:1
        node = f.nodes[k]
        # Storage index if scalar
        j = last(_storage_range(f.sizes, k))
        if node.type == NODE_VARIABLE
            f.forward_storage[j] = x[node.index]
            # This should never happen, because we will have replaced these by now.
            # elseif node.type == Nonlinear.NODE_MOI_VARIABLE
            #     f.forward_storage[k] = x[node.index]
        elseif node.type == NODE_VALUE
            f.forward_storage[j] = f.const_values[node.index]
        elseif node.type == NODE_SUBEXPRESSION
            f.forward_storage[j] = d.subexpression_forward_values[node.index]
        elseif node.type == NODE_PARAMETER
            f.forward_storage[j] = d.data.parameters[node.index]
        elseif node.type == NODE_CALL_MULTIVARIATE
            children_indices = SparseArrays.nzrange(f.adj, k)
            N = length(children_indices)
            # TODO(odow);
            # With appropriate benchmarking, the special-cased if-statements can
            # be removed in favor of the generic user-defined function case.
            if node.index == 1 # :+
                for j in _eachindex(f.sizes, k)
                    tmp_sum = zero(T)
                    for c_idx in children_indices
                        ix = children_arr[c_idx]
                        @j f.partials_storage[ix] = one(T)
                        tmp_sum += @j f.forward_storage[ix]
                    end
                    @j f.forward_storage[k] = tmp_sum
                end
            elseif node.index == 2 # :-
                @assert N == 2
                child1 = first(children_indices)
                @inbounds ix1 = children_arr[child1]
                @inbounds ix2 = children_arr[child1+1]
                for j in _eachindex(f.sizes, k)
                    tmp_sub = @j f.forward_storage[ix1]
                    tmp_sub -= @j f.forward_storage[ix2]
                    @j f.partials_storage[ix1] = one(T)
                    @j f.partials_storage[ix2] = -one(T)
                    @j f.forward_storage[k] = tmp_sub
                end
            elseif node.index == 3 # :*
                # Node `k` is not scalar, so we do matrix multiplication
                if f.sizes.ndims[k] != 0
                    @assert N == 2
                    idx1 = first(children_indices)
                    idx2 = last(children_indices)
                    @inbounds ix1 = children_arr[idx1]
                    @inbounds ix2 = children_arr[idx2]
                    v1 = _view_matrix(f.forward_storage, f.sizes, ix1)
                    v2 = _view_matrix(f.forward_storage, f.sizes, ix2)
                    out = _view_matrix(f.forward_storage, f.sizes, k)
                    LinearAlgebra.mul!(out, v1, v2)
                    # We deliberately don't write v1/v2 into partials_storage
                    # here: the matmul reverse branch reads forward_storage
                    # directly, so those writes were dead.
                    # Node `k` is scalar
                else
                    tmp_prod = one(T)
                    for c_idx in children_indices
                        @inbounds tmp_prod *=
                            f.forward_storage[children_arr[c_idx]]
                    end
                    if tmp_prod == zero(T) || N <= 2
                        # This is inefficient if there are a lot of children.
                        # 2 is chosen as a limit because (x*y)/y does not always
                        # equal x for floating-point numbers. This can produce
                        # unexpected error in partials. There's still an error when
                        # multiplying three or more terms, but users are less likely
                        # to complain about it.
                        for c_idx in children_indices
                            prod_others = one(T)
                            for c_idx2 in children_indices
                                (c_idx == c_idx2) && continue
                                ix = children_arr[c_idx2]
                                prod_others *= f.forward_storage[ix]
                            end
                            f.partials_storage[children_arr[c_idx]] =
                                prod_others
                        end
                    else
                        # Compute all-minus-one partial derivatives by dividing from
                        # the total product.
                        for c_idx in children_indices
                            ix = children_arr[c_idx]
                            f.partials_storage[ix] =
                                tmp_prod / f.forward_storage[ix]
                        end
                    end
                    @inbounds f.forward_storage[k] = tmp_prod
                end
            elseif node.index == 4 # :^
                @assert N == 2
                idx1 = first(children_indices)
                idx2 = last(children_indices)
                @inbounds ix1 = children_arr[idx1]
                @inbounds ix2 = children_arr[idx2]
                @inbounds base = f.forward_storage[ix1]
                @inbounds exponent = f.forward_storage[ix2]
                if exponent == 2
                    @inbounds f.forward_storage[k] = base * base
                    @inbounds f.partials_storage[ix1] = 2 * base
                elseif exponent == 1
                    @inbounds f.forward_storage[k] = base
                    @inbounds f.partials_storage[ix1] = 1.0
                else
                    f.forward_storage[k] = pow(base, exponent)
                    f.partials_storage[ix1] = exponent * pow(base, exponent - 1)
                end
                f.partials_storage[ix2] = f.forward_storage[k] * log(base)
            elseif node.index == 5 # :/
                @assert N == 2
                idx1 = first(children_indices)
                idx2 = last(children_indices)
                @inbounds ix1 = children_arr[idx1]
                @inbounds ix2 = children_arr[idx2]
                @inbounds numerator = f.forward_storage[ix1]
                @inbounds denominator = f.forward_storage[ix2]
                recip_denominator = 1 / denominator
                @inbounds f.partials_storage[ix1] = recip_denominator
                f.partials_storage[ix2] =
                    -numerator * recip_denominator * recip_denominator
                f.forward_storage[k] = numerator * recip_denominator
            elseif node.index == 6 # ifelse
                @assert N == 3
                idx1 = first(children_indices)
                @inbounds condition = f.forward_storage[children_arr[idx1]]
                @inbounds lhs = f.forward_storage[children_arr[idx1+1]]
                @inbounds rhs = f.forward_storage[children_arr[idx1+2]]
                @inbounds f.partials_storage[children_arr[idx1+1]] =
                    condition == 1
                @inbounds f.partials_storage[children_arr[idx1+2]] =
                    !(condition == 1)
                f.forward_storage[k] = ifelse(condition == 1, lhs, rhs)
            elseif node.index == 10 # vect
                for j in _eachindex(f.sizes, k)
                    ix = children_arr[children_indices[j]]
                    @s f.partials_storage[ix] = one(T)
                    val = @s f.forward_storage[ix]
                    @j f.forward_storage[k] = val
                end
            elseif node.index == 11 # dot
                idx1, idx2 = children_indices
                ix1 = children_arr[idx1]
                ix2 = children_arr[idx2]
                tmp_dot = zero(T)
                for j in _eachindex(f.sizes, ix1)
                    v1 = @j f.forward_storage[ix1]
                    v2 = @j f.forward_storage[ix2]
                    @j f.partials_storage[ix1] = v2
                    @j f.partials_storage[ix2] = v1
                    tmp_dot += v1 * v2
                end
                @s f.forward_storage[k] = tmp_dot
            elseif node.index == 12 # hcat
                idx1, idx2 = children_indices
                ix1 = children_arr[idx1]
                ix2 = children_arr[idx2]
                nb_cols1 = f.sizes.ndims[ix1] <= 1 ? 1 : _size(f.sizes, ix1, 2)
                col_size = f.sizes.ndims[ix1] == 0 ? 1 : _size(f.sizes, k, 1)
                for j in _eachindex(f.sizes, ix1)
                    @j f.partials_storage[ix1] = one(T)
                    val = @j f.forward_storage[ix1]
                    @j f.forward_storage[k] = val
                end
                for j in _eachindex(f.sizes, ix2)
                    @j f.partials_storage[ix2] = one(T)
                    val = @j f.forward_storage[ix2]
                    _setindex!(
                        f.forward_storage,
                        val,
                        f.sizes,
                        k,
                        j + nb_cols1 * col_size,
                    )
                end
            elseif node.index == 13 # vcat
                idx1, idx2 = children_indices
                ix1 = children_arr[idx1]
                ix2 = children_arr[idx2]
                nb_rows1 = f.sizes.ndims[ix1] <= 1 ? 1 : _size(f.sizes, ix1, 1)
                nb_rows2 = f.sizes.ndims[ix2] <= 1 ? 1 : _size(f.sizes, ix2, 1)
                nb_rows = nb_rows1 + nb_rows2
                for j in _eachindex(f.sizes, ix1)
                    @j f.partials_storage[ix1] = one(T)
                    val = @j f.forward_storage[ix1]
                    _setindex!(
                        f.forward_storage,
                        val,
                        f.sizes,
                        k,
                        div(j-1, nb_rows1) * nb_rows + 1 + (j-1) % nb_rows1,
                    )
                end
                for j in _eachindex(f.sizes, ix2)
                    @j f.partials_storage[ix2] = one(T)
                    val = @j f.forward_storage[ix2]
                    _setindex!(
                        f.forward_storage,
                        val,
                        f.sizes,
                        k,
                        div(j-1, nb_rows1) * nb_rows +
                        1 +
                        (j-1) % nb_rows1 +
                        nb_rows1,
                    )
                end
            elseif node.index == 14 # norm 
                ix = children_arr[children_indices[1]]
                tmp_norm_squared = zero(T)
                for j in _eachindex(f.sizes, ix)
                    v = @j f.forward_storage[ix]
                    tmp_norm_squared += v * v
                end
                @s f.forward_storage[k] = sqrt(tmp_norm_squared)
                for j in _eachindex(f.sizes, ix)
                    v = @j f.forward_storage[ix]
                    if tmp_norm_squared == 0
                        @j f.partials_storage[ix] = zero(T)
                    else
                        @j f.partials_storage[ix] = v / @s f.forward_storage[k]
                    end
                end
            elseif node.index == 15 # sum
                @assert N == 1
                ix = children_arr[first(children_indices)]
                inp = _view_linear(f.forward_storage, f.sizes, ix)
                fill!(_view_linear(f.partials_storage, f.sizes, ix), one(T))
                @s f.forward_storage[k] = sum(inp)
            elseif node.index == 16 # row
                for j in _eachindex(f.sizes, k)
                    ix = children_arr[children_indices[j]]
                    @s f.partials_storage[ix] = one(T)
                    val = @s f.forward_storage[ix]
                    @j f.forward_storage[k] = val
                end
            else # atan, min, max
                f_input = _UnsafeVectorView(d.jac_storage, N)
                ∇f = _UnsafeVectorView(d.user_output_buffer, N)
                for (r, i) in enumerate(children_indices)
                    f_input[r] = f.forward_storage[children_arr[i]]
                    ∇f[r] = 0.0
                end
                f.forward_storage[k] = eval_multivariate_function(
                    operators,
                    operators.multivariate_operators[node.index],
                    f_input,
                )
                eval_multivariate_gradient(
                    operators,
                    operators.multivariate_operators[node.index],
                    ∇f,
                    f_input,
                )
                for (r, i) in enumerate(children_indices)
                    f.partials_storage[children_arr[i]] = ∇f[r]
                end
            end
        elseif node.type == NODE_CALL_MULTIVARIATE_BROADCASTED
            children_indices = SparseArrays.nzrange(f.adj, k)
            N = length(children_indices)
            if node.index == 1 # :+  (broadcasted)
                for j in _eachindex(f.sizes, k)
                    tmp_sum = zero(T)
                    for c_idx in children_indices
                        ix = children_arr[c_idx]
                        @j f.partials_storage[ix] = one(T)
                        tmp_sum += @j f.forward_storage[ix]
                    end
                    @j f.forward_storage[k] = tmp_sum
                end
            elseif node.index == 2 # :-  (broadcasted)
                @assert N == 2
                child1 = first(children_indices)
                @inbounds ix1 = children_arr[child1]
                @inbounds ix2 = children_arr[child1+1]
                out = _view_linear(f.forward_storage, f.sizes, k)
                v1 = _view_linear(f.forward_storage, f.sizes, ix1)
                v2 = _view_linear(f.forward_storage, f.sizes, ix2)
                out .= v1 .- v2
                fill!(_view_linear(f.partials_storage, f.sizes, ix1), one(T))
                fill!(_view_linear(f.partials_storage, f.sizes, ix2), -one(T))
            elseif node.index == 3 # :*  (broadcasted)
                # Node `k` is not scalar, so we do matrix multiplication
                if f.sizes.ndims[k] != 0
                    @assert N == 2
                    idx1 = first(children_indices)
                    idx2 = last(children_indices)
                    @inbounds ix1 = children_arr[idx1]
                    @inbounds ix2 = children_arr[idx2]
                    v1 = zeros(_size(f.sizes, ix1)...)
                    v2 = zeros(_size(f.sizes, ix2)...)
                    for j in _eachindex(f.sizes, ix1)
                        v1[j] = @j f.forward_storage[ix1]
                        @j f.partials_storage[ix2] = v1[j]
                    end
                    for j in _eachindex(f.sizes, ix2)
                        v2[j] = @j f.forward_storage[ix2]
                        @j f.partials_storage[ix1] = v2[j]
                    end
                    for j in _eachindex(f.sizes, k)
                        @j f.forward_storage[k] = v1[j] * v2[j]
                    end
                    # Node `k` is scalar
                else
                    tmp_prod = one(T)
                    for c_idx in children_indices
                        @inbounds tmp_prod *=
                            f.forward_storage[children_arr[c_idx]]
                    end
                    if tmp_prod == zero(T) || N <= 2
                        # This is inefficient if there are a lot of children.
                        # 2 is chosen as a limit because (x*y)/y does not always
                        # equal x for floating-point numbers. This can produce
                        # unexpected error in partials. There's still an error when
                        # multiplying three or more terms, but users are less likely
                        # to complain about it.
                        for c_idx in children_indices
                            prod_others = one(T)
                            for c_idx2 in children_indices
                                (c_idx == c_idx2) && continue
                                ix = children_arr[c_idx2]
                                prod_others *= f.forward_storage[ix]
                            end
                            f.partials_storage[children_arr[c_idx]] =
                                prod_others
                        end
                    else
                        # Compute all-minus-one partial derivatives by dividing from
                        # the total product.
                        for c_idx in children_indices
                            ix = children_arr[c_idx]
                            f.partials_storage[ix] =
                                tmp_prod / f.forward_storage[ix]
                        end
                    end
                    @inbounds f.forward_storage[k] = tmp_prod
                end
            elseif node.index == 4 # :^ (broadcasted), array .^ scalar
                @assert N == 2
                idx1 = first(children_indices)
                idx2 = last(children_indices)
                @inbounds ix1 = children_arr[idx1]
                @inbounds ix2 = children_arr[idx2]
                @assert f.sizes.ndims[ix2] == 0 "Broadcasted ^ requires scalar exponent"
                exponent = _scalar_load(
                    f.forward_storage,
                    f.sizes.storage_offset[ix2]+1,
                )
                out = _view_linear(f.forward_storage, f.sizes, k)
                inp = _view_linear(f.forward_storage, f.sizes, ix1)
                partials = _view_linear(f.partials_storage, f.sizes, ix1)
                if exponent == 2
                    out .= inp .* inp
                    partials .= 2 .* inp
                elseif exponent == 1
                    out .= inp
                    fill!(partials, one(T))
                else
                    out .= pow.(inp, exponent)
                    partials .= exponent .* pow.(inp, exponent - 1)
                end
            end
        elseif node.type == NODE_CALL_UNIVARIATE
            child_idx = children_arr[f.adj.colptr[k]]
            if node.index == 1 # :+
                for j in _eachindex(f.sizes, k)
                    @j f.partials_storage[child_idx] = one(T)
                    val = @j f.forward_storage[child_idx]
                    @j f.forward_storage[k] = val
                end
            elseif node.index == 2 # :-
                for j in _eachindex(f.sizes, k)
                    @j f.partials_storage[child_idx] = -one(T)
                    val = @j f.forward_storage[child_idx]
                    @j f.forward_storage[k] = -val
                end
            else
                ret_f, ret_f′ = eval_univariate_function_and_gradient(
                    operators,
                    node.index,
                    f.forward_storage[child_idx],
                )
                f.forward_storage[k] = ret_f
                f.partials_storage[child_idx] = ret_f′
            end
        elseif node.type == NODE_CALL_UNIVARIATE_BROADCASTED
            child_idx = children_arr[f.adj.colptr[k]]
            if node.index == 1 # :+
                for j in _eachindex(f.sizes, k)
                    @j f.partials_storage[child_idx] = one(T)
                    val = @j f.forward_storage[child_idx]
                    @j f.forward_storage[k] = val
                end
            elseif node.index == 2 # :-
                for j in _eachindex(f.sizes, k)
                    @j f.partials_storage[child_idx] = -one(T)
                    val = @j f.forward_storage[child_idx]
                    @j f.forward_storage[k] = -val
                end
            elseif operators.univariate_operators[node.index] === :tanh
                out = _view_linear(f.forward_storage, f.sizes, k)
                inp = _view_linear(f.forward_storage, f.sizes, child_idx)
                partials = _view_linear(f.partials_storage, f.sizes, child_idx)
                out .= tanh.(inp)
                partials .= one(T) .- out .* out
            else
                for j in _eachindex(f.sizes, k)
                    ret_f, ret_f′ = eval_univariate_function_and_gradient(
                        operators,
                        node.index,
                        @j f.forward_storage[child_idx]
                    )
                    @j f.forward_storage[k] = ret_f
                    @j f.partials_storage[child_idx] = ret_f′
                end
            end
        elseif node.type == NODE_COMPARISON
            children_idx = SparseArrays.nzrange(f.adj, k)
            result = true
            f.partials_storage[children_arr[children_idx[1]]] = zero(T)
            for r in 2:length(children_idx)
                lhs = children_arr[children_idx[r-1]]
                rhs = children_arr[children_idx[r]]
                result &= eval_comparison_function(
                    operators,
                    operators.comparison_operators[node.index],
                    f.forward_storage[lhs],
                    f.forward_storage[rhs],
                )
                f.partials_storage[rhs] = zero(T)
            end
            f.forward_storage[k] = result
        else
            @assert node.type == NODE_LOGIC
            children_idx = SparseArrays.nzrange(f.adj, k)
            lhs = children_arr[children_idx[1]]
            rhs = children_arr[children_idx[2]]
            f.forward_storage[k] = eval_logic_function(
                operators,
                operators.logic_operators[node.index],
                f.forward_storage[lhs] == 1,
                f.forward_storage[rhs] == 1,
            )
            f.partials_storage[lhs] = zero(T)
            f.partials_storage[rhs] = zero(T)
        end
    end
    # Caller is responsible for reading the right range of `f.forward_storage`
    # for vector-valued roots (use `_storage_range(f.sizes, 1)`); the scalar
    # return is only meaningful when the root is scalar.
    return f.forward_storage[1]
end

"""
    _reverse_eval(f::_SubexpressionStorage)

Reverse-mode evaluation of an expression tree given in `f`.

 * This function assumes `f.partials_storage` is already updated.
 * This function assumes that `f.reverse_storage` has been initialized with 0.0.
"""
function _reverse_eval(
    f::_SubexpressionStorage,
    seed::Union{Nothing,AbstractVector{Float64}} = nothing,
)
    @assert length(f.reverse_storage) >= _length(f.sizes)
    @assert length(f.partials_storage) >= _length(f.sizes)
    # f.nodes is already in order such that parents always appear before
    # children so a forward pass through nodes is a backwards pass through the
    # tree.
    children_arr = SparseArrays.rowvals(f.adj)
    root_range = _storage_range(f.sizes, 1)
    if seed === nothing
        for i in root_range
            f.reverse_storage[i] = one(Float64)
        end
    else
        @assert length(seed) == length(root_range)
        for (j, i) in enumerate(root_range)
            f.reverse_storage[i] = seed[j]
        end
    end
    for k in 1:length(f.nodes)
        node = f.nodes[k]
        children_indices = SparseArrays.nzrange(f.adj, k)
        if node.type == NODE_CALL_MULTIVARIATE
            if node.index in eachindex(DEFAULT_MULTIVARIATE_OPERATORS)
                op = DEFAULT_MULTIVARIATE_OPERATORS[node.index]
                if op == :*
                    if f.sizes.ndims[k] != 0
                        # Matrix multiplication: rev_v1 = rev_parent * v2',
                        # rev_v2 = v1' * rev_parent. Both v1 and v2 are read
                        # straight from forward_storage (the matmul forward
                        # branch deliberately doesn't snapshot them into
                        # partials_storage), and the reverse views are written
                        # in place.
                        idx1 = first(children_indices)
                        idx2 = last(children_indices)
                        ix1 = children_arr[idx1]
                        ix2 = children_arr[idx2]
                        v1 = _view_matrix(f.forward_storage, f.sizes, ix1)
                        v2 = _view_matrix(f.forward_storage, f.sizes, ix2)
                        rev_parent = _view_matrix(f.reverse_storage, f.sizes, k)
                        rev_v1 = _view_matrix(f.reverse_storage, f.sizes, ix1)
                        rev_v2 = _view_matrix(f.reverse_storage, f.sizes, ix2)
                        LinearAlgebra.mul!(rev_v1, rev_parent, v2')
                        LinearAlgebra.mul!(rev_v2, v1', rev_parent)
                        continue
                    end
                elseif op == :vect
                    @assert _eachindex(f.sizes, k) ==
                            eachindex(children_indices)
                    for j in eachindex(children_indices)
                        ix = children_arr[children_indices[j]]
                        rev_parent_j = @j f.reverse_storage[k]
                        # partial is 1 so we can ignore it
                        @s f.reverse_storage[ix] = rev_parent_j
                    end
                    continue
                elseif op == :dot
                    # Node `k` is scalar, the jacobian w.r.t. each vectorized input
                    # child is a row vector whose entries are stored in `f.partials_storage`
                    rev_parent = @s f.reverse_storage[k]
                    for j in
                        _eachindex(f.sizes, children_arr[children_indices[1]])
                        for child_idx in children_indices
                            ix = children_arr[child_idx]
                            partial = @j f.partials_storage[ix]
                            val = ifelse(
                                rev_parent == 0.0 && !isfinite(partial),
                                rev_parent,
                                rev_parent * partial,
                            )
                            @j f.reverse_storage[ix] = val
                        end
                    end
                    continue
                elseif op == :hcat
                    idx1, idx2 = children_indices
                    ix1 = children_arr[idx1]
                    ix2 = children_arr[idx2]
                    nb_cols1 =
                        f.sizes.ndims[ix1] <= 1 ? 1 : _size(f.sizes, ix1, 2)
                    col_size =
                        f.sizes.ndims[ix1] == 0 ? 1 : _size(f.sizes, k, 1)
                    for j in _eachindex(f.sizes, ix1)
                        partial = @j f.partials_storage[ix1]
                        val = ifelse(
                            _getindex(f.reverse_storage, f.sizes, k, j) ==
                            0.0 && !isfinite(partial),
                            _getindex(f.reverse_storage, f.sizes, k, j),
                            _getindex(f.reverse_storage, f.sizes, k, j) *
                            partial,
                        )
                        @j f.reverse_storage[ix1] = val
                    end
                    for j in _eachindex(f.sizes, ix2)
                        partial = @j f.partials_storage[ix2]
                        val = ifelse(
                            _getindex(
                                f.reverse_storage,
                                f.sizes,
                                k,
                                j + nb_cols1 * col_size,
                            ) == 0.0 && !isfinite(partial),
                            _getindex(
                                f.reverse_storage,
                                f.sizes,
                                k,
                                j + nb_cols1 * col_size,
                            ),
                            _getindex(
                                f.reverse_storage,
                                f.sizes,
                                k,
                                j + nb_cols1 * col_size,
                            ) * partial,
                        )
                        @j f.reverse_storage[ix2] = val
                    end
                    continue
                elseif op == :vcat
                    idx1, idx2 = children_indices
                    ix1 = children_arr[idx1]
                    ix2 = children_arr[idx2]
                    nb_rows1 =
                        f.sizes.ndims[ix1] <= 1 ? 1 : _size(f.sizes, ix1, 1)
                    nb_rows2 =
                        f.sizes.ndims[ix2] <= 1 ? 1 : _size(f.sizes, ix2, 1)
                    nb_rows = nb_rows1 + nb_rows2
                    row_size =
                        f.sizes.ndims[ix1] == 0 ? 1 : _size(f.sizes, k, 2)
                    for j in _eachindex(f.sizes, ix1)
                        partial = @j f.partials_storage[ix1]
                        val = ifelse(
                            _getindex(
                                f.reverse_storage,
                                f.sizes,
                                k,
                                div(j-1, nb_rows1) * nb_rows +
                                1 +
                                (j-1) % nb_rows1,
                            ) == 0.0 && !isfinite(partial),
                            _getindex(
                                f.reverse_storage,
                                f.sizes,
                                k,
                                div(j-1, nb_rows1) * nb_rows +
                                1 +
                                (j-1) % nb_rows1,
                            ),
                            _getindex(
                                f.reverse_storage,
                                f.sizes,
                                k,
                                div(j-1, nb_rows1) * nb_rows +
                                1 +
                                (j-1) % nb_rows1,
                            ) * partial,
                        )
                        @j f.reverse_storage[ix1] = val
                    end
                    for j in _eachindex(f.sizes, ix2)
                        partial = @j f.partials_storage[ix2]
                        val = ifelse(
                            _getindex(
                                f.reverse_storage,
                                f.sizes,
                                k,
                                div(j-1, nb_rows1) * nb_rows +
                                1 +
                                (j-1) % nb_rows1 +
                                nb_rows1,
                            ) == 0.0 && !isfinite(partial),
                            _getindex(
                                f.reverse_storage,
                                f.sizes,
                                k,
                                div(j-1, nb_rows1) * nb_rows +
                                1 +
                                (j-1) % nb_rows1 +
                                nb_rows1,
                            ),
                            _getindex(
                                f.reverse_storage,
                                f.sizes,
                                k,
                                div(j-1, nb_rows1) * nb_rows +
                                1 +
                                (j-1) % nb_rows1 +
                                nb_rows1,
                            ) * partial,
                        )
                        @j f.reverse_storage[ix2] = val
                    end
                    continue
                elseif op == :norm
                    # Node `k` is scalar, the jacobian w.r.t. the vectorized input
                    # child is a row vector whose entries are stored in `f.partials_storage`
                    rev_parent = @s f.reverse_storage[k]
                    for j in
                        _eachindex(f.sizes, children_arr[children_indices[1]])
                        ix = children_arr[children_indices[1]]
                        partial = @j f.partials_storage[ix]
                        val = ifelse(
                            rev_parent == 0.0 && !isfinite(partial),
                            rev_parent,
                            rev_parent * partial,
                        )
                        @j f.reverse_storage[ix] = val
                    end
                    continue
                elseif op == :sum
                    # `sum` is rank-reducing (1 → 0): reverse-mode broadcasts
                    # the parent's scalar adjoint to every child slot.
                    ix = children_arr[children_indices[1]]
                    pos = _scalar_pos(f.sizes, k)
                    # Avoid the scalar read of `reverse_storage[k]` (which fails
                    # on # GPU storage) by indexing with a 0-dim index, the view
                    # is then 0-dim at the outermost type, which the
                    # broadcast machinery specializes as a scalar source.
                    rev_parent_view =
                        view(f.reverse_storage, reshape(pos:pos, ()))
                    rev_children_view =
                        _view_linear(f.reverse_storage, f.sizes, ix)
                    # On GPU this lowers to a single fill-kernel; no
                    # Device-to-Host round-trip.
                    rev_children_view .= rev_parent_view
                    continue
                elseif op == :row
                    for j in _eachindex(f.sizes, k)
                        ix = children_arr[children_indices[j]]
                        rev_parent_j = @j f.reverse_storage[k]
                        # partial is 1 so we can ignore it
                        @s f.reverse_storage[ix] = rev_parent_j
                    end
                    continue
                end
            end
        elseif node.type == NODE_CALL_MULTIVARIATE_BROADCASTED
            if node.index in eachindex(DEFAULT_MULTIVARIATE_OPERATORS)
                op = DEFAULT_MULTIVARIATE_OPERATORS[node.index]
                if op == :*
                    if f.sizes.ndims[k] != 0
                        # Node `k` is not scalar, so we do matrix multiplication or broadcasted multiplication
                        idx1 = first(children_indices)
                        idx2 = last(children_indices)
                        ix1 = children_arr[idx1]
                        ix2 = children_arr[idx2]
                        v1 = zeros(_size(f.sizes, ix1)...)
                        v2 = zeros(_size(f.sizes, ix2)...)
                        for j in _eachindex(f.sizes, ix1)
                            v1[j] = @j f.forward_storage[ix1]
                        end
                        for j in _eachindex(f.sizes, ix2)
                            v2[j] = @j f.forward_storage[ix2]
                        end
                        rev_parent = zeros(_size(f.sizes, k)...)
                        for j in _eachindex(f.sizes, k)
                            rev_parent[j] = @j f.reverse_storage[k]
                        end
                        rev_v1 = zeros(_size(f.sizes, ix1)...)
                        rev_v2 = zeros(_size(f.sizes, ix2)...)
                        for j in _eachindex(f.sizes, ix1)
                            rev_v1[j] = rev_parent[j] * v2[j]
                            @j f.reverse_storage[ix1] = rev_v1[j]
                        end
                        for j in _eachindex(f.sizes, ix2)
                            rev_v2[j] = rev_parent[j] * v1[j]
                            @j f.reverse_storage[ix2] = rev_v2[j]
                        end
                        continue
                    end
                elseif op == :^
                    # Broadcasted array .^ scalar: vectorize the per-element
                    # base reverse (with 0*Inf guard preserved) and reduce
                    # the exponent contribution as a single `sum` over GPU
                    # arrays.
                    @assert length(children_indices) == 2
                    idx1 = first(children_indices)
                    idx2 = last(children_indices)
                    @inbounds ix1 = children_arr[idx1]
                    @inbounds ix2 = children_arr[idx2]
                    rev_parent = _view_linear(f.reverse_storage, f.sizes, k)
                    rev_v1 = _view_linear(f.reverse_storage, f.sizes, ix1)
                    partial = _view_linear(f.partials_storage, f.sizes, ix1)
                    rev_v1 .= ifelse.(
                        (rev_parent .== 0) .& .!isfinite.(partial),
                        rev_parent,
                        rev_parent .* partial,
                    )
                    base_view = _view_linear(f.forward_storage, f.sizes, ix1)
                    out_view = _view_linear(f.forward_storage, f.sizes, k)
                    # `mapreduce(f, +, base_view, rev_parent, out_view)`
                    # would express this directly, but multi-iterable
                    # `mapreduce` materializes an intermediate today
                    # (JuliaLang/julia#53417). Wrap the inputs in `zip` so
                    # the single-iterable specialization fires and the
                    # reduction stays allocation-free. Once
                    # https://github.com/JuliaLang/julia/pull/55301 lands
                    # we can drop the `zip` and use the multi-arg form.
                    T = eltype(rev_parent)
                    rev_exp_total = mapreduce(
                        +,
                        zip(base_view, rev_parent, out_view);
                        init = zero(T),
                    ) do (b, rp, o)
                        return b > 0 ? rp * o * log(b) : zero(T)
                    end
                    pos2 = _scalar_pos(f.sizes, ix2)
                    view(f.reverse_storage, pos2:pos2) .= rev_exp_total
                    continue
                end
            end
        elseif node.type != NODE_CALL_UNIVARIATE &&
               node.type != NODE_CALL_UNIVARIATE_BROADCASTED
            continue
        end
        # Node `k` has same size as its children.
        # The Jacobian (between the vectorized versions) is diagonal and the
        # diagonal entries are stored in `f.partials_storage`. We broadcast
        # `rev_child .= rev_parent .* partial` over the whole array (with the
        # 0 * Inf guard preserved).
        rev_parent = _view_linear(f.reverse_storage, f.sizes, k)
        for child_idx in children_indices
            ix = children_arr[child_idx]
            @assert _size(f.sizes, k) == _size(f.sizes, ix)
            rev_child = _view_linear(f.reverse_storage, f.sizes, ix)
            partial = _view_linear(f.partials_storage, f.sizes, ix)
            rev_child .= ifelse.(
                (rev_parent .== 0) .& .!isfinite.(partial),
                rev_parent,
                rev_parent .* partial,
            )
        end
    end
    return
end

"""
    _extract_reverse_pass(
        g::AbstractVector{T},
        d::NLPEvaluator,
        f::_FunctionStorage,
    ) where {T}

Fill the gradient vector `g` with the values from the reverse pass. Assumes you
have already called `_reverse_eval_all(d, x)`.
"""
function _extract_reverse_pass(
    g::AbstractVector{T},
    d::NLPEvaluator,
    f::_FunctionStorage,
) where {T}
    for i in f.dependent_subexpressions
        d.subexpression_reverse_values[i] = 0.0
    end
    _extract_reverse_pass_inner(g, f, d.subexpression_reverse_values, 1.0)
    for i in length(f.dependent_subexpressions):-1:1
        k = f.dependent_subexpressions[i]
        _extract_reverse_pass_inner(
            g,
            d.subexpressions[k],
            d.subexpression_reverse_values,
            d.subexpression_reverse_values[k],
        )
    end
    return
end

function _extract_reverse_pass_inner(
    output::AbstractVector{T},
    f::_FunctionStorage,
    subexpressions::AbstractVector{T},
    scale::T,
) where {T}
    return _extract_reverse_pass_inner(output, f.expr, subexpressions, scale)
end

function _extract_reverse_pass_inner(
    output::AbstractVector{T},
    f::_SubexpressionStorage,
    subexpressions::AbstractVector{T},
    scale::T,
) where {T}
    @assert length(f.reverse_storage) >= _length(f.sizes)
    for (k, node) in enumerate(f.nodes)
        if node.type == NODE_VARIABLE
            output[node.index] += scale * @s f.reverse_storage[k]
        elseif node.type == NODE_SUBEXPRESSION
            subexpressions[node.index] += scale * @s f.reverse_storage[k]
        end
    end
    return
end
