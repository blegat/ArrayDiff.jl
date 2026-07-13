# CPU timing of one AL-gradient evaluation (all residual groups: forward +
# J'v, plus the objective gradient) for the polar form, comparing the
# structured GatherMatrix constants against SparseMatrixCSC.
#
#   julia --project=. bench_cpu.jl [case118.m]

include("ACOPF.jl")

import Downloads
import Printf

# Fetch a pglib-opf case into /tmp (cached across runs).
function pglib_case(name::AbstractString)
    path = joinpath(tempdir(), name)
    if !isfile(path)
        Downloads.download(
            "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3/$name",
            path,
        )
    end
    return path
end

function bench_case(case)
    path = startswith(case, "pglib") ? pglib_case(case) : matpower_case(case)
    d = parse_polar_case(path)
    println("$case: $(d.nbus) buses, $(d.nbranch) branches, $(d.ngen) gens")
    for (label, kw) in [
        ("GatherMatrix", (use_gather = true,)),
        ("SparseMatrixCSC", (use_gather = false,)),
    ]
        prob = build_polar(d; kw...)
        st = ALState(prob)
        al_gradient!(st, prob, 10.0) # compile
        n = 1_000
        t = @elapsed for _ in 1:n
            al_gradient!(st, prob, 10.0)
        end
        Printf.@printf("  %-16s %8.1f µs / AL gradient\n", label, 1e6 * t / n)
    end
    return
end

for case in (
    isempty(ARGS) ?
    [
        "case9.m",
        "pglib_opf_case118_ieee.m",
        "pglib_opf_case1354_pegase.m",
    ] : ARGS
)
    bench_case(case)
end
