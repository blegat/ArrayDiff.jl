# Needs https://github.com/jump-dev/JuMP.jl/pull/3451
using JuMP

include(joinpath(@__DIR__, "array_of_variables.jl"))
include(joinpath(@__DIR__, "array_expr.jl"))

n = 2
X = rand(n, n)
model = Model()
@variable(model, W[1:n, 1:n], container = ArrayOfVariables)
W * X
tanh.(W * X)
