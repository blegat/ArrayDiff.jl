module TestEvalResidualGPU

using Test

using JuMP
using ArrayDiff
import CUDA
import MathOptInterface as MOI

function runtests()
    if !CUDA.functional()
        @info "CUDA is not functional in this environment; skipping GPU tests."
        return
    end
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

# Regression test for the branch's `_read_residual!` change in
# `src/mathoptinterface_api.jl`. The old element-wise loop
#     for (i, j) in enumerate(range)
#         F[i] = res.expr.forward_storage[j]
#     end
# triggers a scalar-indexing error when either `F` or `forward_storage` lives
# on the GPU. The new implementation uses `copyto!` with views, which dispatches
# to `cudaMemcpy` for same-dtype CuArray ↔ CuArray (or CuArray ↔ contiguous
# CPU buffer) transfers.
function _residual_fn(W1, b1, W2, b2)
    return x -> W2 * tanh.(W1 * x .+ b1) .+ b2
end

function test_eval_residual_gpu_matches_cpu()
    # Small two-layer MLP residual: 3 → 4 → 2.
    W1 = [0.4 -0.2 0.1; -0.3 0.5 0.2; 0.1 0.1 -0.4; 0.2 -0.1 0.3]
    b1 = [0.05, -0.1, 0.1, 0.0]
    W2 = [0.3 -0.4 0.2 0.1; -0.1 0.2 0.3 -0.5]
    b2 = [0.0, 0.0]
    f = _residual_fn(W1, b1, W2, b2)
    input_dim = 3
    output_dim = 2
    x_cpu = [0.6, -0.3, 0.4]
    expected = f(x_cpu)
    # CPU evaluator as a reference.
    cpu_eval = ArrayDiff.evaluator(f, input_dim)
    F_cpu = zeros(Float64, output_dim)
    ArrayDiff.eval_residual!(cpu_eval, F_cpu, x_cpu)
    @test F_cpu ≈ expected
    # GPU evaluator: forward_storage lives on the device. `_read_residual!`
    # must copy `forward_storage::CuVector → F::CuVector` without scalar
    # indexing.
    gpu_eval = ArrayDiff.evaluator(
        f,
        input_dim;
        mode = ArrayDiff.Mode{CUDA.CuVector{Float64}}(),
    )
    F_gpu = CUDA.zeros(Float64, output_dim)
    x_gpu = CUDA.CuVector{Float64}(x_cpu)
    ArrayDiff.eval_residual!(gpu_eval, F_gpu, x_gpu)
    @test Array(F_gpu) ≈ expected
    return
end

end

TestEvalResidualGPU.runtests()
