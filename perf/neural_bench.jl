include("lux.jl")
include("hand_cuda.jl")
include("pytorch.jl")
include("arraydiff.jl")

function _display(name, bench)
    println("### $name")
    println("```")
    display(bench)
    println("```")
    return
end

function compare(T, h, d, n; gpu)
    _display("Lux", LuxNeural.neural(T, h, d, n; gpu))
    _display(
        "Hand-CUDA without prealloc",
        HandCuda.neural(T, h, d, n; prealloc = false, gpu),
    )
    _display(
        "Hand-CUDA with prealloc",
        HandCuda.neural(T, h, d, n; prealloc = true, gpu),
    )
    _display(
        "PyTorch eager",
        PyTorchNeural.neural(T, h, d, n; eager = true, gpu),
    )
    _display(
        "PyTorch compiled",
        PyTorchNeural.neural(T, h, d, n; eager = false, gpu),
    )
    _display("ArrayDiff", ArrayDiffNeural.neural(T, h, d, n; gpu))
    return
end

T, h, d, n = Float32, 4096, 13, 178
println("## CPU")
compare(T, h, d, n; gpu = false)
println("## GPU")
compare(T, h, d, n; gpu = true)
