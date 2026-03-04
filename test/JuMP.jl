module TestJuMP

using Test

using JuMP
using ArrayDiff

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

function test_array_product()
    n = 2
    X = rand(n, n)
    model = Model()
    @variable(model, W[1:n, 1:n], container = ArrayDiff.ArrayOfVariables)
    @test W isa ArrayDiff.MatrixOfVariables{Float64}
    @test JuMP.index(W[1, 1]) == MOI.VariableIndex(1)
    @test JuMP.index(W[2, 1]) == MOI.VariableIndex(2)
    @test JuMP.index(W[2]) == MOI.VariableIndex(2)
    @test sprint(show, W) ==
          "2×2 ArrayDiff.ArrayOfVariables{Float64, 2} with offset 0"
    prod = W * X
    @test prod isa ArrayDiff.ArrayExpr{2}
    @test sprint(show, prod) ==
          "2×2 ArrayDiff.GenericArrayExpr{JuMP.VariableRef, 2}"
    err = ErrorException(
        "`getindex` not implemented, build vectorized expression instead",
    )
    @test_throws err prod[1, 1]
    return
end

end  # module

TestJuMP.runtests()
