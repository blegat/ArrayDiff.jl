using BenchmarkTools, Random

function compares(a)
    s = 0
    for i in eachindex(a)
        for j in eachindex(a)
            s += isless(a[i], a[j])
        end
    end
    return s
end

time_compares(a) = @time compares(a)

num = 1000

time_compares(rand(Int, num))

using Random

function bench_symbols(len, num)
    a = [Symbol(randstring(len)) for _ in 1:num]
    time_compares(a)
    time_compares(a)
end

bench_symbols(3, 1000)

function bench_strings(len, num)
    a = [randstring(len) for _ in 1:num]
    time_compares(a)
    time_compares(a)
end

bench_strings(3, 1000)

op = :tanh
f = eval(op)
f(1)
ChainRulesCore.rrule(f, x)
