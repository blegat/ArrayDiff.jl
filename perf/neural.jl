# Neural network optimization using ArrayDiff + NLopt
#
# This demonstrates end-to-end optimization of a simple two-layer neural
# network with array-valued decision variables, array-aware AD, and a
# first-order NLP solver.

using JuMP
using ArrayDiff
using LinearAlgebra
import NLopt

n = 2
X = rand(n, n)
target = rand(n, n)

model = Model(NLopt.Optimizer)
set_attribute(model, "algorithm", :LD_LBFGS)

@variable(model, W1[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
@variable(model, W2[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)

# Set non-zero starting values to avoid saddle point at zero
for i in 1:n, j in 1:n
    set_start_value(W1[i, j], 0.1 * randn())
    set_start_value(W2[i, j], 0.1 * randn())
end

# Forward pass: Y = W2 * tanh.(W1 * X)
Y = W2 * tanh.(W1 * X)

# Loss: ||Y - target||  (norm returns a scalar NonlinearExpr)
# Pre-compute expression before @objective to avoid macro rewriting of `.-`
loss = norm(Y .- target)
@objective(model, Min, loss)

optimize!(model)

println("Termination status: ", termination_status(model))
println("Objective value:    ", objective_value(model))
println("W1 = ", [value(W1[i, j]) for i in 1:n, j in 1:n])
println("W2 = ", [value(W2[i, j]) for i in 1:n, j in 1:n])
