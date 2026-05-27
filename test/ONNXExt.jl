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
    return ONNX.AttributeProto(
        name = name,
        i = Int64(i),
        var"#type" = AT.INT,
    )
end

function _attr_tensor(name::String, t::ONNX.TensorProto)
    return ONNX.AttributeProto(
        name = name,
        t = t,
        var"#type" = AT.TENSOR,
    )
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
    W = [1.0 0.5;
         -0.2 1.1;
         0.3 -0.7]
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
    W = [0.4 -0.6 0.2;
         1.1 0.3 -0.9]  # (2, 3)
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
    fjulia(x) = sum((α .* (reshape(x, 1, 2) * W) .+ β .* reshape(bias, 1, 3)) .^ 2)
    @test val ≈ fjulia(xv)
    @test g ≈ ForwardDiff.gradient(fjulia, xv)
end

# Gemm with transB=1: y = X * W'  (PyTorch nn.Linear export pattern).
# W is stored as (out, in); Gemm transposes it. X is shape (1, in).
function test_gemm_transB()
    vars = [MOI.VariableIndex(i) for i in 1:2]
    var_mat = reshape(vars, 1, 2)
    W = [0.4 0.5;
         -0.1 1.0;
         0.9 -0.3]  # (3, 2) — Linear(in=2, out=3)
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
    n1 = _make_node("Constant", String[], ["c"]; attrs = [_attr_tensor("value", const_t)], name = "k")
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
function test_mlp_relu()
    D_in, D_hidden, D_out = 3, 4, 2
    vars = [MOI.VariableIndex(i) for i in 1:D_in]
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
        _make_node("Gemm", ["x", "W1", "b1"], ["h_pre"]; attrs = gemm_attrs(1), name = "fc1"),
        _make_node("Relu", ["h_pre"], ["h"]; name = "act"),
        _make_node("Gemm", ["h", "W2", "b2"], ["y"]; attrs = gemm_attrs(1), name = "fc2"),
    ]
    proto = _build_model(nodes, ["x"], ["y"]; initializers = init)
    xv = [0.5, -0.7, 1.1]
    val, g = _eval_with_gradient(proto, vars, xv)
    fjulia(x) = begin
        h = max.(x' * W1' .+ b1', 0.0)  # row vec
        y = h * W2' .+ b2'
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
    proto = _build_model(
        [_make_node("Identity", ["x"], ["y"])],
        ["x"],
        ["y"],
    )
    @test_throws ErrorException ArrayDiff.from_onnx(proto; inputs = Dict())
end

end # module

TestONNXExt.runtests()
