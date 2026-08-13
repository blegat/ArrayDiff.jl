include("arraydiff.jl")
T, h, d, n = Float32, 4096, 13, 178
display(ArrayDiffNeural.neural(T, h, d, n; gpu = true))
display(ArrayDiffNeural.profile_gpu())
