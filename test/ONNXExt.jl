module TestONNXExt

using Test
import LinearAlgebra
import ForwardDiff
import MathOptInterface as MOI

import ArrayDiff
import ONNX  # loads ArrayDiffONNXExt

const AT = getfield(ONNX, Symbol("AttributeProto.AttributeType"))
const DT = getfield(ONNX, Symbol("TensorProto.DataType"))

# ── ONNX protobuf builder helpers (kept in-test; not part of the package API) ─

# ONNX serializes row-major; Julia is column-major. Mirror the inverse of the
# loader's reshape so a Julia `Matrix` round-trips through (build → load).
function _flatten_row_major(A::AbstractArray)
    if ndims(A) <= 1
        return Float64[Float64(x) for x in A]
    elseif ndims(A) == 2
        return Float64[Float64(A[i, j]) for i in axes(A, 1) for j in axes(A, 2)]
    else
        error("Only 1D/2D tensors supported in tests")
    end
end

function _make_tensor(name::String, data::AbstractArray)
    return ONNX.TensorProto(
        dims = Int64[size(data)...],
        data_type = Int32(DT.DOUBLE),
        name = name,
        double_data = _flatten_row_major(data),
    )
end

function _make_scalar_tensor(name::String, v::Real)
    return ONNX.TensorProto(
        dims = Int64[],
        data_type = Int32(DT.DOUBLE),
        name = name,
        double_data = Float64[Float64(v)],
    )
end

function _attr_float(name::String, f::Real)
    return ONNX.AttributeProto(
        name = name,
        f = Float32(f),
        var"#type" = AT.FLOAT,
    )
end

function _attr_int(name::String, i::Integer)
    return ONNX.AttributeProto(name = name, i = Int64(i), var"#type" = AT.INT)
end

function _attr_tensor(name::String, t::ONNX.TensorProto)
    return ONNX.AttributeProto(name = name, t = t, var"#type" = AT.TENSOR)
end

function _make_node(
    op_type::String,
    inputs::Vector{String},
    outputs::Vector{String};
    attrs::AbstractVector = ONNX.AttributeProto[],
    name::String = "n",
)
    # `AttributeProto` is parametric; an array literal of one concrete kind
    # narrows its eltype, so re-wrap into the UnionAll-eltype vector that
    # `NodeProto` expects.
    attribute = ONNX.AttributeProto[a for a in attrs]
    return ONNX.NodeProto(
        input = inputs,
        output = outputs,
        name = name,
        op_type = op_type,
        domain = "",
        attribute = attribute,
        doc_string = "",
    )
end

function _vinfo(name::String)
    return ONNX.ValueInfoProto(
        name = name,
        var"#type" = nothing,
        doc_string = "",
    )
end

function _build_model(
    nodes::Vector{ONNX.NodeProto},
    inputs::Vector{String},
    outputs::Vector{String};
    initializers::Vector{ONNX.TensorProto} = ONNX.TensorProto[],
)
    g = ONNX.GraphProto(
        node = nodes,
        name = "g",
        initializer = initializers,
        input = ONNX.ValueInfoProto[_vinfo(n) for n in inputs],
        output = ONNX.ValueInfoProto[_vinfo(n) for n in outputs],
    )
    return ONNX.ModelProto(
        ir_version = Int64(7),
        producer_name = "test",
        producer_version = "0",
        domain = "",
        model_version = Int64(0),
        doc_string = "",
        graph = g,
    )
end

# ── Evaluation harness ───────────────────────────────────────────────────────

# Build an ArrayDiff model with objective = scalar_fn(from_onnx(proto)), then
# return (value, gradient) at `xv`. `vars` are the input variables that get
# bound to the graph's single input "x".
function _eval_with_gradient(
    proto::ONNX.ModelProto,
    vars::Vector{MOI.VariableIndex},
    xv::Vector{Float64};
    input = vars,  # what gets bound to ONNX input "x" (Vector or Matrix of vars)
    scalar_fn::Symbol = :dot,  # :dot(out, out) — sum of squares
)
    out = ArrayDiff.from_onnx(proto; inputs = Dict("x" => input))
    snf = if scalar_fn === :dot
        MOI.ScalarNonlinearFunction(:dot, Any[out, out])
    elseif scalar_fn === :sum
        MOI.ScalarNonlinearFunction(:sum, Any[out])
    else
        error("unknown scalar_fn")
    end
    model = ArrayDiff.Model()
    ArrayDiff.set_objective(model, snf)
    evaluator = ArrayDiff.Evaluator(model, ArrayDiff.Mode(), vars)
    MOI.initialize(evaluator, [:Grad])
    val = MOI.eval_objective(evaluator, xv)
    g = zeros(length(xv))
    MOI.eval_objective_gradient(evaluator, g, xv)
    return val, g
end

# ── Tests ────────────────────────────────────────────────────────────────────

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

# Identity: y = x; ‖y‖² = ‖x‖² → gradient = 2x.
function test_identity()
    n = 3
    vars = [MOI.VariableIndex(i) for i in 1:n]
    node = _make_node("Identity", ["x"], ["y"])
    proto = _build_model([node], ["x"], ["y"])
    xv = [0.5, -1.2, 2.0]
    val, g = _eval_with_gradient(proto, vars, xv)
    @test val ≈ sum(xv .^ 2)
    @test g ≈ 2 .* xv
end

# Add with a constant vector bias: y = x .+ b
function test_add_constant_bias()
    n = 4
    vars = [MOI.VariableIndex(i) for i in 1:n]
    b = [0.1, -0.3, 0.7, 1.0]
    init = _make_tensor("b", b)
    node = _make_node("Add", ["x", "b"], ["y"])
    proto = _build_model([node], ["x"], ["y"]; initializers = [init])
    xv = [1.0, 2.0, -1.5, 0.4]
    val, g = _eval_with_gradient(proto, vars, xv)
    fjulia(x) = sum((x .+ b) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Sub / Mul / Div: cover the rest of the broadcasted-elementwise family.
function test_elementwise_sub_mul_div()
    n = 3
    vars = [MOI.VariableIndex(i) for i in 1:n]
    c = [2.0, -1.0, 0.5]
    init = _make_tensor("c", c)
    for (op, fjl) in [
        ("Sub", (x) -> sum((x .- c) .^ 2)),
        ("Mul", (x) -> sum((x .* c) .^ 2)),
        ("Div", (x) -> sum((x ./ c) .^ 2)),
    ]
        node = _make_node(op, ["x", "c"], ["y"])
        proto = _build_model([node], ["x"], ["y"]; initializers = [init])
        xv = [0.7, 1.3, -0.4]
        val, g = _eval_with_gradient(proto, vars, xv)
        @test val ≈ fjl(xv)
        @test g ≈ ForwardDiff.gradient(fjl, xv)
    end
end

# MatMul vector × matrix: y = x * W, W is (3, 2). Output is 1D length 2.
function test_matmul_vector_matrix()
    vars = [MOI.VariableIndex(i) for i in 1:3]
    W = [
        1.0 0.5;
        -0.2 1.1;
        0.3 -0.7
    ]
    init = _make_tensor("W", W)
    node = _make_node("MatMul", ["x", "W"], ["y"])
    proto = _build_model([node], ["x"], ["y"]; initializers = [init])
    xv = [0.4, -1.0, 0.9]
    val, g = _eval_with_gradient(proto, vars, xv)
    fjulia(x) = sum((x' * W) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Gemm without transB: y = α * (X * W) + β * b. X is shape (1, K).
function test_gemm_no_transpose()
    vars = [MOI.VariableIndex(i) for i in 1:2]
    var_mat = reshape(vars, 1, 2)  # (1, 2)
    W = [
        0.4 -0.6 0.2;
        1.1 0.3 -0.9
    ]  # (2, 3)
    bias = [0.05, -0.1, 0.2]
    α, β = 0.5, 2.0
    init_W = _make_tensor("W", W)
    init_b = _make_tensor("b", bias)
    node = _make_node(
        "Gemm",
        ["x", "W", "b"],
        ["y"];
        attrs = [
            _attr_float("alpha", α),
            _attr_float("beta", β),
            _attr_int("transA", 0),
            _attr_int("transB", 0),
        ],
    )
    proto = _build_model([node], ["x"], ["y"]; initializers = [init_W, init_b])
    xv = [0.8, -0.3]
    val, g = _eval_with_gradient(proto, vars, xv; input = var_mat)
    fjulia(x) =
        sum((α .* (reshape(x, 1, 2) * W) .+ β .* reshape(bias, 1, 3)) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Gemm with transB=1: y = X * W'  (PyTorch nn.Linear export pattern).
# W is stored as (out, in); Gemm transposes it. X is shape (1, in).
function test_gemm_transB()
    vars = [MOI.VariableIndex(i) for i in 1:2]
    var_mat = reshape(vars, 1, 2)
    W = [
        0.4 0.5;
        -0.1 1.0;
        0.9 -0.3
    ]  # (3, 2) — Linear(in=2, out=3)
    bias = [0.0, 0.0, 0.0]
    init_W = _make_tensor("W", W)
    init_b = _make_tensor("b", bias)
    node = _make_node(
        "Gemm",
        ["x", "W", "b"],
        ["y"];
        attrs = [
            _attr_float("alpha", 1.0),
            _attr_float("beta", 1.0),
            _attr_int("transA", 0),
            _attr_int("transB", 1),
        ],
    )
    proto = _build_model([node], ["x"], ["y"]; initializers = [init_W, init_b])
    xv = [1.1, -0.4]
    val, g = _eval_with_gradient(proto, vars, xv; input = var_mat)
    fjulia(x) = sum((reshape(x, 1, 2) * W' .+ reshape(bias, 1, 3)) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Relu broadcast: y = max.(x, 0); gradient passes through positive entries.
function test_relu()
    vars = [MOI.VariableIndex(i) for i in 1:4]
    node = _make_node("Relu", ["x"], ["y"])
    proto = _build_model([node], ["x"], ["y"])
    xv = [1.0, -0.5, 0.3, -2.0]
    val, g = _eval_with_gradient(proto, vars, xv)
    fjulia(x) = sum(max.(x, 0.0) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

function test_tanh()
    vars = [MOI.VariableIndex(i) for i in 1:3]
    node = _make_node("Tanh", ["x"], ["y"])
    proto = _build_model([node], ["x"], ["y"])
    xv = [0.2, -0.8, 1.5]
    val, g = _eval_with_gradient(proto, vars, xv)
    fjulia(x) = sum(tanh.(x) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

function test_sigmoid()
    vars = [MOI.VariableIndex(i) for i in 1:3]
    node = _make_node("Sigmoid", ["x"], ["y"])
    proto = _build_model([node], ["x"], ["y"])
    xv = [0.1, -1.2, 0.7]
    val, g = _eval_with_gradient(proto, vars, xv)
    σ(t) = 1 / (1 + exp(-t))
    fjulia(x) = sum(σ.(x) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Constant op: y = Identity(Constant_value)*0 + x. Just check the value path.
function test_constant_then_add()
    vars = [MOI.VariableIndex(i) for i in 1:3]
    c = [0.5, -0.5, 1.0]
    const_t = _make_tensor("c_value", c)
    n1 = _make_node(
        "Constant",
        String[],
        ["c"];
        attrs = [_attr_tensor("value", const_t)],
        name = "k",
    )
    n2 = _make_node("Add", ["x", "c"], ["y"])
    proto = _build_model([n1, n2], ["x"], ["y"])
    xv = [0.3, 0.8, -0.4]
    val, g = _eval_with_gradient(proto, vars, xv)
    fjulia(x) = sum((x .+ c) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# End-to-end: a 1-hidden-layer MLP with relu, matching what a PyTorch
# `nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, D_out))` would export.
# Input is shape (1, D_in), matching the (batch, features) convention.
function test_mlp_relu()
    D_in, D_hidden, D_out = 3, 4, 2
    vars = [MOI.VariableIndex(i) for i in 1:D_in]
    var_mat = reshape(vars, 1, D_in)
    W1 = 0.3 * randn(MersenneTwisterRNG(), D_hidden, D_in)  # (H, D_in)
    b1 = 0.1 * randn(MersenneTwisterRNG(2), D_hidden)
    W2 = 0.5 * randn(MersenneTwisterRNG(3), D_out, D_hidden)  # (D_out, H)
    b2 = 0.2 * randn(MersenneTwisterRNG(4), D_out)
    init = [
        _make_tensor("W1", W1),
        _make_tensor("b1", b1),
        _make_tensor("W2", W2),
        _make_tensor("b2", b2),
    ]
    gemm_attrs(transB) = [
        _attr_float("alpha", 1.0),
        _attr_float("beta", 1.0),
        _attr_int("transA", 0),
        _attr_int("transB", transB),
    ]
    nodes = [
        _make_node(
            "Gemm",
            ["x", "W1", "b1"],
            ["h_pre"];
            attrs = gemm_attrs(1),
            name = "fc1",
        ),
        _make_node("Relu", ["h_pre"], ["h"]; name = "act"),
        _make_node(
            "Gemm",
            ["h", "W2", "b2"],
            ["y"];
            attrs = gemm_attrs(1),
            name = "fc2",
        ),
    ]
    proto = _build_model(nodes, ["x"], ["y"]; initializers = init)
    xv = [0.5, -0.7, 1.1]
    val, g = _eval_with_gradient(proto, vars, xv; input = var_mat)
    fjulia(x) = begin
        xrow = reshape(x, 1, D_in)
        h = max.(xrow * W1' .+ reshape(b1, 1, D_hidden), 0.0)
        y = h * W2' .+ reshape(b2, 1, D_out)
        sum(y .^ 2)
    end
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Tiny RNG helper so the test is reproducible without needing the user's
# global RNG state.
import Random
MersenneTwisterRNG(seed::Int = 1) = Random.MersenneTwister(seed)

# Error paths.
function test_unsupported_op_errors()
    vars = [MOI.VariableIndex(i) for i in 1:2]
    node = _make_node("LeakyRelu", ["x"], ["y"])
    proto = _build_model([node], ["x"], ["y"])
    @test_throws ErrorException ArrayDiff.from_onnx(
        proto;
        inputs = Dict("x" => vars),
    )
end

function test_missing_input_errors()
    proto = _build_model([_make_node("Identity", ["x"], ["y"])], ["x"], ["y"])
    @test_throws ErrorException ArrayDiff.from_onnx(proto; inputs = Dict())
end

# ── Direct tests for private helpers ─────────────────────────────────────────

const _ext = Base.get_extension(ArrayDiff, :ArrayDiffONNXExt)

function test_broadcast_shape_helper()
    @test _ext._broadcast_shape((), (3,)) == (3,)
    @test _ext._broadcast_shape((3,), ()) == (3,)
    @test _ext._broadcast_shape((2, 3), (2, 3)) == (2, 3)
    # da == 1 broadcast
    @test _ext._broadcast_shape((1, 3), (2, 3)) == (2, 3)
    # db == 1 broadcast (1D bias-style)
    @test _ext._broadcast_shape((2, 3), (3,)) == (2, 3)
    # incompatible
    @test_throws ErrorException _ext._broadcast_shape((2,), (3,))
end

function test_wrap_input_scalar_vector_matrix_real()
    @test _ext._wrap_input(3.5) == (3.5, ())
    @test _ext._wrap_input(2) == (2.0, ())
    v = [1.0, 2.0, 3.0]
    out, sz = _ext._wrap_input(v)
    @test out == v && sz == (3,) && out isa Vector{Float64}
    M = [1.0 2.0; 3.0 4.0]
    outM, szM = _ext._wrap_input(M)
    @test outM == M && szM == (2, 2) && outM isa Matrix{Float64}
end

function test_wrap_input_anf()
    anf = ArrayDiff.ArrayNonlinearFunction{1}(:vect, Any[1.0, 2.0], (2,), false)
    out, sz = _ext._wrap_input(anf)
    @test out === anf && sz == (2,)
end

function test_wrap_input_unsupported()
    @test_throws ErrorException _ext._wrap_input((1, 2, 3))
end

function test_wrap_input_matrix_vars_multi_row()
    M = collect(reshape([MOI.VariableIndex(i) for i in 1:6], 2, 3))
    out, sz = _ext._wrap_input(M)
    @test sz == (2, 3)
    @test out isa ArrayDiff.ArrayNonlinearFunction{2}
    @test out.head == :vcat
end

# ── TensorProto encoding paths ───────────────────────────────────────────────

function test_tensor_to_array_float_data()
    t = ONNX.TensorProto(
        dims = Int64[2, 3],
        data_type = Int32(DT.FLOAT),
        name = "t",
        float_data = Float32[1, 2, 3, 4, 5, 6],
    )
    arr, sz = _ext._tensor_to_array(t)
    @test sz == (2, 3)
    @test arr == [1.0 2.0 3.0; 4.0 5.0 6.0]
end

function test_tensor_to_array_raw_data_float()
    raw = Vector{UInt8}(reinterpret(UInt8, Float32[1.0, 2.0, 3.0]))
    t = ONNX.TensorProto(
        dims = Int64[3],
        data_type = Int32(DT.FLOAT),
        name = "t",
        raw_data = raw,
    )
    arr, sz = _ext._tensor_to_array(t)
    @test sz == (3,)
    @test arr == [1.0, 2.0, 3.0]
end

function test_tensor_to_array_raw_data_double()
    raw = Vector{UInt8}(reinterpret(UInt8, Float64[1.5, -2.5]))
    t = ONNX.TensorProto(
        dims = Int64[2],
        data_type = Int32(DT.DOUBLE),
        name = "t",
        raw_data = raw,
    )
    arr, sz = _ext._tensor_to_array(t)
    @test sz == (2,)
    @test arr == [1.5, -2.5]
end

function test_tensor_to_array_raw_data_unsupported()
    t = ONNX.TensorProto(
        dims = Int64[2],
        data_type = Int32(DT.INT32),
        name = "t",
        raw_data = UInt8[1, 2, 3, 4, 5, 6, 7, 8],
    )
    @test_throws ErrorException _ext._tensor_to_array(t)
end

function test_tensor_to_array_empty_encoding()
    t = ONNX.TensorProto(
        dims = Int64[2],
        data_type = Int32(DT.INT32),
        name = "t",
    )
    @test_throws ErrorException _ext._tensor_to_array(t)
end

function test_tensor_to_array_scalar()
    t = _make_scalar_tensor("t", 3.5)
    arr, sz = _ext._tensor_to_array(t)
    @test arr == 3.5 && sz == ()
end

function test_tensor_to_array_3d_unsupported()
    t = ONNX.TensorProto(
        dims = Int64[1, 2, 3],
        data_type = Int32(DT.DOUBLE),
        name = "t",
        double_data = Float64[1, 2, 3, 4, 5, 6],
    )
    @test_throws ErrorException _ext._tensor_to_array(t)
end

# ── Per-op coverage ──────────────────────────────────────────────────────────

function test_neg()
    vars = [MOI.VariableIndex(i) for i in 1:3]
    node = _make_node("Neg", ["x"], ["y"])
    proto = _build_model([node], ["x"], ["y"])
    xv = [0.4, -1.2, 1.7]
    val, g = _eval_with_gradient(proto, vars, xv)
    fjulia(x) = sum((-x) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Constant op with a scalar value.
function test_constant_scalar()
    vars = [MOI.VariableIndex(i) for i in 1:2]
    c_t = _make_scalar_tensor("c_val", 1.5)
    n1 = _make_node(
        "Constant",
        String[],
        ["c"];
        attrs = [_attr_tensor("value", c_t)],
        name = "k",
    )
    n2 = _make_node("Add", ["x", "c"], ["y"])
    proto = _build_model([n1, n2], ["x"], ["y"])
    xv = [0.5, 1.0]
    val, g = _eval_with_gradient(proto, vars, xv)
    fjulia(x) = sum((x .+ 1.5) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

function test_constant_missing_value_errors()
    n = _make_node("Constant", String[], ["c"])
    proto = _build_model(
        [n, _make_node("Identity", ["c"], ["y"])],
        String[],
        ["y"],
    )
    @test_throws ErrorException ArrayDiff.from_onnx(proto)
end

# MatMul Mat × Vec: y = X * b, X = (2, 3) vars, b = (3,) const.
function test_matmul_matrix_vector()
    vars = [MOI.VariableIndex(i) for i in 1:6]
    var_mat = collect(reshape(vars, 2, 3))
    b = [0.4, -1.0, 0.9]
    init = _make_tensor("b", b)
    node = _make_node("MatMul", ["x", "b"], ["y"])
    proto = _build_model([node], ["x"], ["y"]; initializers = [init])
    xv = [0.3, -0.7, 1.1, 2.0, 0.5, -1.5]
    val, g = _eval_with_gradient(proto, vars, xv; input = var_mat)
    fjulia(x) = sum((reshape(x, 2, 3) * b) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# MatMul Mat × Mat: y = X * W, X = (2, 3) vars, W = (3, 2) const.
function test_matmul_matrix_matrix()
    vars = [MOI.VariableIndex(i) for i in 1:6]
    var_mat = collect(reshape(vars, 2, 3))
    W = [
        0.4 -0.1
        0.5 1.2
        -0.3 0.7
    ]
    init = _make_tensor("W", W)
    node = _make_node("MatMul", ["x", "W"], ["y"])
    proto = _build_model([node], ["x"], ["y"]; initializers = [init])
    xv = [0.2, 1.0, -0.5, 0.8, 1.4, -1.1]
    val, g = _eval_with_gradient(proto, vars, xv; input = var_mat)
    fjulia(x) = sum((reshape(x, 2, 3) * W) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Vec × Vec is not a supported MatMul shape combination.
function test_matmul_unsupported_shapes()
    vars = [MOI.VariableIndex(i) for i in 1:3]
    b = [1.0, 2.0, 3.0]
    init = _make_tensor("b", b)
    node = _make_node("MatMul", ["x", "b"], ["y"])
    proto = _build_model([node], ["x"], ["y"]; initializers = [init])
    @test_throws ErrorException ArrayDiff.from_onnx(
        proto;
        inputs = Dict("x" => vars),
    )
end

# Gemm without C: 2-input form, no bias.
function test_gemm_no_bias()
    vars = [MOI.VariableIndex(i) for i in 1:2]
    var_mat = reshape(vars, 1, 2)
    W = [
        0.4 -0.1
        0.5 1.2
    ]
    init_W = _make_tensor("W", W)
    node = _make_node(
        "Gemm",
        ["x", "W"],
        ["y"];
        attrs = [
            _attr_float("alpha", 1.0),
            _attr_float("beta", 1.0),
            _attr_int("transA", 0),
            _attr_int("transB", 0),
        ],
    )
    proto = _build_model([node], ["x"], ["y"]; initializers = [init_W])
    xv = [0.3, -0.7]
    val, g = _eval_with_gradient(proto, vars, xv; input = var_mat)
    fjulia(x) = sum((reshape(x, 1, 2) * W) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Gemm with all attributes omitted: covers the default-attr path in `_find_attr`.
function test_gemm_default_attrs()
    vars = [MOI.VariableIndex(i) for i in 1:2]
    var_mat = reshape(vars, 1, 2)
    W = [
        0.4 -0.1
        0.5 1.2
    ]
    bias = [0.1, -0.2]
    init_W = _make_tensor("W", W)
    init_b = _make_tensor("b", bias)
    node = _make_node("Gemm", ["x", "W", "b"], ["y"])
    proto = _build_model([node], ["x"], ["y"]; initializers = [init_W, init_b])
    xv = [0.3, -0.7]
    val, g = _eval_with_gradient(proto, vars, xv; input = var_mat)
    fjulia(x) = sum((reshape(x, 1, 2) * W .+ reshape(bias, 1, 2)) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Gemm with a non-constant 1D bias is rejected.
function test_gemm_non_const_1d_bias_errors()
    vars_x = [MOI.VariableIndex(i) for i in 1:2]
    vars_c = [MOI.VariableIndex(i) for i in 3:5]
    var_mat = reshape(vars_x, 1, 2)
    W = [
        0.4 -0.1 0.5
        1.2 -0.3 0.7
    ]
    init_W = _make_tensor("W", W)
    node = _make_node("Gemm", ["x", "W", "c"], ["y"])
    proto = _build_model([node], ["x", "c"], ["y"]; initializers = [init_W])
    @test_throws ErrorException ArrayDiff.from_onnx(
        proto;
        inputs = Dict("x" => var_mat, "c" => vars_c),
    )
end

# Graph declares "x" as both an input and an initializer: the initializer wins.
function test_input_name_overlaps_initializer()
    init_x = _make_tensor("x", [1.0, 2.0, 3.0])
    node = _make_node("Identity", ["x"], ["y"])
    proto = _build_model([node], ["x"], ["y"]; initializers = [init_x])
    out = ArrayDiff.from_onnx(proto)
    @test out == [1.0, 2.0, 3.0]
end

# Multi-output graph: result is keyed by output name.
function test_multi_output()
    vars = [MOI.VariableIndex(i) for i in 1:3]
    n1 = _make_node("Identity", ["x"], ["a"]; name = "id")
    n2 = _make_node("Neg", ["x"], ["b"]; name = "neg")
    proto = _build_model([n1, n2], ["x"], ["a", "b"])
    out = ArrayDiff.from_onnx(proto; inputs = Dict("x" => vars))
    @test out isa Dict
    @test sort(collect(keys(out))) == ["a", "b"]
end

end # module

TestONNXExt.runtests()
