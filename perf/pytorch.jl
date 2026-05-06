module PyTorchNeural

# PyTorch autograd of the 2-layer MLP loss
#
#   L = sum((W2 * tanh(W1 * X) - y) .^ 2)
#
# `neural(T, h, d, n; eager, gpu)` builds tensors + warms up the function
# (paying any `torch.compile` codegen cost), then returns a
# `BenchmarkTools.Trial` that times only the per-call forward+backward.

using Random
using BenchmarkTools
using PythonCall

const torch = pyimport("torch")
const np = pyimport("numpy")

# Output dimension `m`. Fixed at 2 to match the other modules.
const OUT_DIM = 2

const _grad_fn_eager, _grad_fn_compiled = let
    # `import torch` inside `_eager` so it lands in the function's
    # `__globals__` at call time — `@pyexec` runs with separate globals and
    # locals dicts, and the top-level `import torch` only populates locals,
    # leaving `_eager` unable to resolve `torch` when invoked later.
    #
    # `_compiled = torch.compile(_eager)` runs Inductor codegen on the first
    # call for each new input shape (multi-second cost); subsequent calls
    # reuse the cached graph.
    nt = @pyexec """
import torch
def _eager(W1, W2, X, y):
    import torch
    y1 = torch.tanh(torch.matmul(W1, X))
    diff = torch.matmul(W2, y1) - y
    loss = (diff * diff).sum()
    return torch.autograd.grad(loss, W1)[0]
_compiled = torch.compile(_eager)
""" => (_eager::Py, _compiled::Py)
    (nt._eager, nt._compiled)
end

const _torch_dtype =
    Dict{DataType,Py}(Float32 => torch.float32, Float64 => torch.float64)

# Julia is column-major, NumPy is row-major. `PyArray(::Matrix)` exposes the
# same memory as an F-contiguous numpy view; `np.ascontiguousarray` copies it
# into C-contiguous layout so `torch.from_numpy` gets the shape PyTorch
# expects without surprises.
function _to_device(arr::AbstractArray, dtype::Py, device::String)
    np_arr = np.ascontiguousarray(np.asarray(PyArray(arr)))
    return torch.from_numpy(np_arr).to(device, dtype = dtype)
end

"""
    neural(T, h, d, n; eager::Bool = true, gpu::Bool = false) -> BenchmarkTools.Trial

Benchmark `∂L/∂W1` of the 2-layer MLP loss using PyTorch autograd. Tensor
setup and the warmup call (which pays `torch.compile`'s codegen if
`eager=false`) happen once before the trial; the timed block is just the
per-call forward+backward plus a `cuda.synchronize` when `gpu=true`.

`eager = true` (default) uses standard eager-mode autograd. `eager = false`
runs the same function under `torch.compile`, which fuses elementwise ops
into Triton kernels.
"""
function neural(
    ::Type{T},
    h::Int,
    d::Int,
    n::Int;
    eager::Bool = true,
    gpu::Bool = false,
) where {T<:Real}
    Random.seed!(0)
    W1 = randn(T, h, d)
    W2 = randn(T, OUT_DIM, h)
    X = randn(T, d, n)
    y = randn(T, OUT_DIM, n)
    dt = _torch_dtype[T]
    device = gpu ? "cuda" : "cpu"
    W1t = _to_device(W1, dt, device).requires_grad_(true)
    W2t = _to_device(W2, dt, device)
    Xt = _to_device(X, dt, device)
    yt = _to_device(y, dt, device)
    fn = eager ? _grad_fn_eager : _grad_fn_compiled
    fn(W1t, W2t, Xt, yt)  # warmup (codegen for compiled path)
    if gpu
        torch.cuda.synchronize()
    end
    return @benchmark begin
        $fn($W1t, $W2t, $Xt, $yt)
        if $gpu
            $torch.cuda.synchronize()
        end
    end
end

end # module
