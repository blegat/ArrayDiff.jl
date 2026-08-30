# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    print_tree([io::IO=stdout], expr::Expression; operators=OperatorRegistry())

Print the expression graph rooted at `expr` as a unicode tree.

`operators` is used to resolve operator and registered-operator names. For an
expression that lives in a `Model`, pass `model.operators` so user-registered
operators render with their registered names rather than numeric indices.
"""
function print_tree end

print_tree(expr::Expression; kwargs...) = print_tree(stdout, expr; kwargs...)

function print_tree(
    io::IO,
    expr::Expression;
    operators::OperatorRegistry = OperatorRegistry(),
)
    if isempty(expr.nodes)
        println(io, "<empty expression>")
        return
    end
    root = findfirst(n -> n.parent <= 0, expr.nodes)
    @assert root !== nothing "Expression has no root node"
    adj = adjacency_matrix(expr.nodes)
    children_arr = SparseArrays.rowvals(adj)
    children_of(k) =
        sort!(collect(view(children_arr, SparseArrays.nzrange(adj, k))))
    println(io, _node_label(expr, operators, root))
    _print_children(io, expr, operators, children_of, root, "")
    return
end

function _print_children(io, expr, operators, children_of, k, prefix)
    kids = children_of(k)
    n = length(kids)
    for (i, c) in enumerate(kids)
        last = (i == n)
        branch = last ? "└─ " : "├─ "
        cont = last ? "   " : "│  "
        println(io, prefix, branch, _node_label(expr, operators, c))
        _print_children(io, expr, operators, children_of, c, prefix * cont)
    end
    return
end

function _node_label(expr::Expression, operators::OperatorRegistry, k::Int)
    node = expr.nodes[k]
    t = node.type
    if t == NODE_CALL_MULTIVARIATE
        return string(_op_name(operators.multivariate_operators, node.index))
    elseif t == NODE_CALL_UNIVARIATE
        return string(_op_name(operators.univariate_operators, node.index))
    elseif t == NODE_CALL_MULTIVARIATE_BROADCASTED
        return string(
            _op_name(operators.multivariate_operators, node.index),
            ".",
        )
    elseif t == NODE_CALL_UNIVARIATE_BROADCASTED
        return string(_op_name(operators.univariate_operators, node.index), ".")
    elseif t == NODE_CALL_REDUCE
        return string(
            "reduce(",
            _op_name(operators.multivariate_operators, node.index),
            ")",
        )
    elseif t == NODE_LOGIC
        return string(_op_name(operators.logic_operators, node.index))
    elseif t == NODE_COMPARISON
        return string(_op_name(operators.comparison_operators, node.index))
    elseif t == NODE_VARIABLE
        return string("x[", node.index, "]")
    elseif t == NODE_MOI_VARIABLE
        return string("MOI.VariableIndex(", node.index, ")")
    elseif t == NODE_VALUE
        return string(expr.values[node.index])
    elseif t == NODE_PARAMETER
        return string("parameter[", node.index, "]")
    elseif t == NODE_SUBEXPRESSION
        return string("subexpression[", node.index, "]")
    elseif t == NODE_VARIABLE_BLOCK
        return string("x[", node.index, "] ", _shape_str(expr, k))
    elseif t == NODE_MOI_VARIABLE_BLOCK
        return string(
            "MOI.VariableIndex(",
            node.index,
            ") ",
            _shape_str(expr, k),
        )
    elseif t == NODE_VALUE_BLOCK
        shape = get(expr.block_shapes, k, Int[])
        len = isempty(shape) ? 1 : prod(shape)
        return string(
            "values[",
            node.index,
            ":",
            node.index + len - 1,
            "] ",
            _shape_str(shape),
        )
    end
    return string(t, "(", node.index, ")")
end

function _op_name(ops::Vector{Symbol}, idx::Int)
    return 1 <= idx <= length(ops) ? ops[idx] : Symbol("op#", idx)
end

_shape_str(expr::Expression, k::Int) = _shape_str(get(expr.block_shapes, k, Int[]))
_shape_str(shape::Vector{Int}) =
    isempty(shape) ? "" : string("(", join(shape, "×"), ")")

function Base.show(io::IO, ::MIME"text/plain", expr::Expression)
    println(io, "Expression with ", length(expr.nodes), " nodes:")
    print_tree(io, expr)
    return
end
