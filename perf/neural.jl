# Needs https://github.com/jump-dev/JuMP.jl/pull/3451
using JuMP
using ArrayDiff

n = 2
X = rand(n, n)
model = Model()
@variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
W * X
