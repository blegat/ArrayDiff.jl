module ArrayDiffONNXExt

import ArrayDiff
import MathOptInterface as MOI
import ONNX

const ANF = ArrayDiff.ArrayNonlinearFunction
const _Shape = Tuple{Vararg{Int}}
# An entry in the conversion env: the value plus its array shape.
# Value may be `ANF`, `MOI.ScalarNonlinearFunction`, `MOI.VariableIndex`,
# `Float64`, `Vector{Float64}`, or `Matrix{Float64}`. Shape is `()` for scalars.
const _Entry = Tuple{Any,_Shape}

# ── Attribute helpers ───────────────────────────────────────────────────────

function _find_attr(node::ONNX.NodeProto, name::AbstractString)
    for a in node.attribute
        if a.name == name
            return a
        end
    end
    return nothing
end

function _attr_int(node, name; default::Int)
    a = _find_attr(node, name)
    return a === nothing ? default : Int(a.i)
end

function _attr_float(node, name; default::Float64)
    a = _find_attr(node, name)
    return a === nothing ? default : Float64(a.f)
end

function _attr_tensor(node, name)
    a = _find_attr(node, name)
    return a === nothing ? nothing : a.t
end

# ── TensorProto → Julia array ───────────────────────────────────────────────
# ONNX stores tensor values in C (row-major) order. Julia is column-major,
# so a 2D tensor of dims=(m, n) must be reshaped to (n, m) then permuted.

function _tensor_to_array(t::ONNX.TensorProto)
    DT = getfield(ONNX, Symbol("TensorProto.DataType"))
    dims = Int[Int(d) for d in t.dims]
    flat = if t.data_type == Int32(DT.DOUBLE) && !isempty(t.double_data)
        Float64[x for x in t.double_data]
    elseif t.data_type == Int32(DT.FLOAT) && !isempty(t.float_data)
        Float64[Float64(x) for x in t.float_data]
    elseif !isempty(t.raw_data)
        if t.data_type == Int32(DT.FLOAT)
            Float64.(reinterpret(Float32, t.raw_data))
        elseif t.data_type == Int32(DT.DOUBLE)
            Float64.(reinterpret(Float64, t.raw_data))
        else
            error("Unsupported raw_data type: $(t.data_type)")
        end
    else
        error("Unsupported tensor encoding for '$(t.name)'")
    end
    if isempty(dims)
        @assert length(flat) == 1
        return (flat[1], ())
    elseif length(dims) == 1
        return (flat, (dims[1],))
    elseif length(dims) == 2
        m, n = dims
        # ONNX row-major (m, n) → Julia column-major: reshape (n, m), permute.
        mat = collect(permutedims(reshape(flat, (n, m)), (2, 1)))
        return (mat, (m, n))
    else
        error(
            "Tensors with ndim > 2 not supported (got dims=$dims for '$(t.name)')",
        )
    end
end

# ── User-supplied input wrapping ────────────────────────────────────────────

function _wrap_input(v::Vector{MOI.VariableIndex})
    return (ANF{1}(:vect, Any[v...], (length(v),), false), (length(v),))
end

function _wrap_input(M::Matrix{MOI.VariableIndex})
    m, n = size(M)
    row(i) = ANF{2}(:row, Any[M[i, j] for j in 1:n], (1, n), false)
    if m == 1
        return (row(1), (1, n))
    end
    # ArrayDiff's `:vcat` evaluator handles exactly two children (see
    # reverse_mode.jl); fold left so each `:vcat` node stays binary.
    acc = row(1)
    for i in 2:m
        acc = ANF{2}(:vcat, Any[acc, row(i)], (i, n), false)
    end
    return (acc, (m, n))
end

_wrap_input(x::ANF{N}) where {N} = (x, x.size)
_wrap_input(x::Real) = (Float64(x), ())
_wrap_input(x::Vector{<:Real}) = (Vector{Float64}(x), (length(x),))
_wrap_input(x::Matrix{<:Real}) = (Matrix{Float64}(x), size(x))

_wrap_input(x) = error("Unsupported input value type: $(typeof(x))")

# ── Shape arithmetic ────────────────────────────────────────────────────────

function _broadcast_shape(a::_Shape, b::_Shape)
    # NumPy / ONNX semantics: align trailing dims; each pair must match or one is 1.
    if a == ()
        return b
    end
    if b == ()
        return a
    end
    n = max(length(a), length(b))
    out = Vector{Int}(undef, n)
    for i in 1:n
        da = i <= length(a) ? a[end-i+1] : 1
        db = i <= length(b) ? b[end-i+1] : 1
        if da == db
            out[n-i+1] = da
        elseif da == 1
            out[n-i+1] = db
        elseif db == 1
            out[n-i+1] = da
        else
            error("Incompatible broadcast shapes $a vs $b")
        end
    end
    return tuple(out...)
end

# ── Op-emit helpers ─────────────────────────────────────────────────────────

# Broadcasted multivariate op
function _bcall(op::Symbol, args::Vector{Any}, shape::_Shape)
    N = length(shape)
    return ANF{N}(op, args, shape, true)
end

# Non-broadcast multivariate op (used for matmul)
function _call(op::Symbol, args::Vector{Any}, shape::_Shape)
    N = length(shape)
    return ANF{N}(op, args, shape, false)
end

# ── Per-op conversion ───────────────────────────────────────────────────────

function _convert_node(node::ONNX.NodeProto, env::Dict{String,_Entry})
    op = node.op_type
    if op == "Identity"
        return env[node.input[1]]

    elseif op == "Constant"
        t = _attr_tensor(node, "value")
        t === nothing &&
            error("Constant node '$(node.name)' has no 'value' attribute")
        return _tensor_to_array(t)

    elseif op == "Add"
        return _binop_broadcast(:+, node, env)
    elseif op == "Sub"
        return _binop_broadcast(:-, node, env)
    elseif op == "Mul"
        return _binop_broadcast(:*, node, env)
    elseif op == "Div"
        return _binop_broadcast(:/, node, env)

    elseif op == "Neg"
        x, sx = env[node.input[1]]
        return (_bcall(:-, Any[0.0, x], sx), sx)

    elseif op == "MatMul"
        return _convert_matmul(node, env)

    elseif op == "Gemm"
        return _convert_gemm(node, env)

    elseif op == "Relu"
        # ArrayDiff's broadcasted-multivariate shape inference only handles
        # {+, -, *, /, ^}, not `:max`. Express ReLU using the equivalent
        # `(x + abs(x)) / 2` which uses only broadcast-supported ops.
        x, sx = env[node.input[1]]
        absx = _bcall(:abs, Any[x], sx)
        s = _bcall(:+, Any[x, absx], sx)
        return (_bcall(:/, Any[s, 2.0], sx), sx)

    elseif op == "Tanh"
        x, sx = env[node.input[1]]
        return (_bcall(:tanh, Any[x], sx), sx)

    elseif op == "Sigmoid"
        # 1 / (1 + exp(-x)), all broadcast over x's shape.
        x, sx = env[node.input[1]]
        negx = _bcall(:-, Any[0.0, x], sx)
        ex = _bcall(:exp, Any[negx], sx)
        one_plus = _bcall(:+, Any[1.0, ex], sx)
        return (_bcall(:/, Any[1.0, one_plus], sx), sx)

    else
        error("ONNX op '$(op)' is not supported by ArrayDiffONNXExt")
    end
end

function _binop_broadcast(op::Symbol, node, env)
    a, sa = env[node.input[1]]
    b, sb = env[node.input[2]]
    s = _broadcast_shape(sa, sb)
    return (_bcall(op, Any[a, b], s), s)
end

function _convert_matmul(node, env)
    a, sa = env[node.input[1]]
    b, sb = env[node.input[2]]
    if length(sa) == 1 && length(sb) == 2
        # NumPy-style Vec × Mat = Vec. ArrayDiff's `:*` shape inference walks
        # left-to-right and requires the first non-scalar child to be 2D, so
        # rewrite as Matᵀ × Vec by transposing the constant matrix at
        # convert time. Requires `b` to be a constant tensor (initializer).
        sa[1] == sb[1] || error("MatMul shape mismatch: $sa × $sb")
        b isa AbstractMatrix{<:Real} || error(
            "MatMul Vec × Mat requires the matrix to be a constant initializer (got $(typeof(b)))",
        )
        bT = collect(permutedims(b))
        s = (sb[2],)
        return (_call(:*, Any[bT, a], s), s)
    elseif length(sa) == 2 && length(sb) == 1
        sa[2] == sb[1] || error("MatMul shape mismatch: $sa × $sb")
        s = (sa[1],)
        return (_call(:*, Any[a, b], s), s)
    elseif length(sa) == 2 && length(sb) == 2
        sa[2] == sb[1] || error("MatMul shape mismatch: $sa × $sb")
        s = (sa[1], sb[2])
        return (_call(:*, Any[a, b], s), s)
    else
        error("MatMul with shapes $sa, $sb not supported")
    end
end

function _convert_gemm(node, env)
    A, sA = env[node.input[1]]
    B, sB = env[node.input[2]]
    has_C = length(node.input) >= 3 && !isempty(node.input[3])
    C, sC = has_C ? env[node.input[3]] : (nothing, ())
    α = _attr_float(node, "alpha"; default = 1.0)
    β = _attr_float(node, "beta"; default = 1.0)
    transA = _attr_int(node, "transA"; default = 0) != 0
    transB = _attr_int(node, "transB"; default = 0) != 0

    transA && error("Gemm with transA=1 is not supported")
    if transB
        B isa AbstractMatrix{<:Real} || error(
            "Gemm with transB=1 requires B to be a constant tensor (got $(typeof(B)))",
        )
        B = collect(permutedims(B))
        sB = (sB[2], sB[1])
    end

    # ONNX Gemm requires 2D inputs; we follow the spec strictly. The user is
    # expected to wrap a single sample as `Matrix{MOI.VariableIndex}` of shape
    # `(1, K)`, matching the (batch, features) convention PyTorch exports use.
    length(sA) == 2 && length(sB) == 2 ||
        error("Gemm requires 2D inputs (got A=$sA, B=$sB)")
    sA[2] == sB[1] || error("Gemm: A=$sA, B=$sB shape mismatch")
    AB_shape = (sA[1], sB[2])
    AB = _call(:*, Any[A, B], AB_shape)

    if α != 1.0
        AB = _bcall(:*, Any[α, AB], AB_shape)
    end

    if !has_C || β == 0.0
        return (AB, AB_shape)
    end

    # ONNX broadcasts a (N,) bias against (M, N) by treating the bias as a row.
    # ArrayDiff (following Julia) would treat (N,) as a column instead, so
    # promote to (1, N) explicitly before adding.
    if length(sC) == 1
        if C isa AbstractVector{<:Real}
            C = reshape(collect(C), 1, sC[1])
        else
            error("Gemm with non-constant 1D bias is not supported yet")
        end
        sC = (1, sC[1])
    end

    Cterm = β == 1.0 ? C : _bcall(:*, Any[β, C], sC)
    out_shape = _broadcast_shape(AB_shape, sC)
    out = _bcall(:+, Any[AB, Cterm], out_shape)
    return (out, out_shape)
end

# ── Entry point ─────────────────────────────────────────────────────────────

function ArrayDiff.from_onnx(
    proto::ONNX.ModelProto;
    inputs::AbstractDict = Dict{String,Any}(),
)
    graph = proto.graph
    env = Dict{String,_Entry}()

    for tp in graph.initializer
        env[tp.name] = _tensor_to_array(tp)
    end

    for inp in graph.input
        if haskey(env, inp.name)
            continue  # already provided as initializer
        end
        haskey(inputs, String(inp.name)) ||
            error("ONNX graph input '$(inp.name)' has no supplied value")
        env[inp.name] = _wrap_input(inputs[String(inp.name)])
    end

    for node in graph.node
        result = _convert_node(node, env)
        # All currently-supported ops produce exactly one output.
        length(node.output) == 1 ||
            error("Multi-output op '$(node.op_type)' not supported")
        env[node.output[1]] = result
    end

    out_names = [String(o.name) for o in graph.output]
    if length(out_names) == 1
        return env[out_names[1]][1]
    else
        return Dict(name => env[name][1] for name in out_names)
    end
end

end # module
