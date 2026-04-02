# Needs https://github.com/jump-dev/JuMP.jl/pull/3451
using JuMP
using ArrayDiff
import LinearAlgebra

n = 2
X = rand(n, n)
Y = rand(n, n)
model = Model()
@variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
@variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
Y_hat = W2 * tanh.(W1 * X)
loss = LinearAlgebra.norm(Y_hat .- Y)
@objective(model, Min, loss)
