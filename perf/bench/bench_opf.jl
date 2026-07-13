# Apples-to-apples AC-OPF derivative-timing harness.
#
# Three backends produce the *same* AC-OPF as an `NLPModels.AbstractNLPModel`:
#   :exa       — ExaModels' `ac_power_model` (docs/src/opf.jl)          → ExaModel
#   :genopt    — GenOpt's model (examples/opf.jl) → ExaModels.ExaModel  → ExaModel
#   :arraydiff — this repo's vectorized ArrayDiffNLPModel               (perf/percival)
#
# Because all three are `AbstractNLPModel`, the timing step is identical for
# each — same call, gradient vs gradient, residual vs residual.
#
#   julia --project=perf/bench
#   include("perf/bench/bench_opf.jl")
#   m = opf_model(:arraydiff, "case14.m")   # step 1: build (PowerModels-bundled)
#   opf_timings(m)                          # step 2: time
#
# Case names ending up in PowerModels' bundled data ("case9.m"/"case14.m"/
# "case30.m") load offline; pglib names ("pglib_opf_case118_ieee.m") download.
#
# Swap storage/backend for GPU (you take over here): pass `backend=CUDABackend()`
# to :exa/:genopt, or `mode=ArrayDiff.Mode{CuVector{Float64}}(), device=CuArray`
# to :arraydiff.

import ExaModels
import GenOpt
import NLPModels
import Printf
using BenchmarkTools

# `:arraydiff` builder + its deps (parse_polar_case, build_polar_nlp).
include(joinpath(@__DIR__, "..", "percival", "build_percival.jl"))

import Downloads

const PGLIB_URL =
    "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3"

# Resolve a case name to a local `.m` file. Prefer the MATPOWER cases bundled
# with PowerModels (`matpower_case`, from data.jl) — e.g. "case9.m", "case14.m",
# "case30.m" — so no download is needed. Fall back to a cached pglib download
# for names PowerModels doesn't ship (e.g. "pglib_opf_case118_ieee.m").
function case_path(name::AbstractString)
    bundled = matpower_case(name)
    isfile(bundled) && return bundled
    path = joinpath(tempdir(), name)
    isfile(path) || Downloads.download("$(PGLIB_URL)/$(name)", path)
    return path
end

# Substituted into GenOpt's example so it uses the resolved local file instead
# of re-downloading (see the :genopt builder).
_bench_skip_download(args...) = nothing

# Load only the function definitions from a script `path` (everything before the
# `stop` marker), optionally applying `subs` string replacements first. Lets us
# reuse the real upstream opf.jl files without their download/solve tails.
function _include_prefix(path, stop; subs = ())
    src = read(path, String)
    for (a, b) in subs
        src = replace(src, a => b)
    end
    i = findfirst(stop, src)
    i === nothing || (src = src[1:prevind(src, first(i))])
    return Base.include_string(Main, src, path)
end

const EXA_OPF = joinpath(dirname(dirname(pathof(ExaModels))), "docs", "src", "opf.jl")

# ── Builders ─────────────────────────────────────────────────────────────────

function opf_model(::Val{:arraydiff}, name; kwargs...)
    d = parse_polar_case(case_path(name))
    return build_polar_nlp(d; kwargs...)[1]
end

function opf_model(::Val{:exa}, name; backend = nothing, T = Float64)
    isdefined(Main, :ac_power_model) ||
        _include_prefix(EXA_OPF, "# We first download")
    return Base.invokelatest(Main.ac_power_model, case_path(name); backend = backend, T = T)
end

include(joinpath(dirname(dirname(pathof(GenOpt))), "examples", "opf", "model.jl"))

function opf_model(::Val{:genopt}, name; backend = nothing)
    return ExaModels.ExaModel(build_model(PGLib.pglib(name)); backend = backend)
end

opf_model(backend::Symbol, name; kwargs...) =
    opf_model(Val(backend), name; kwargs...)

# ── Timing (uniform NLPModels API) ───────────────────────────────────────────

"""
    opf_timings(nlp) -> NamedTuple

Median wall-clock time of one evaluation of each first-order primitive, at the
model's starting point. `residual` here is the constraint vector `cons!`.
"""
function opf_timings(nlp)
    n, m = nlp.meta.nvar, nlp.meta.ncon
    x = copy(nlp.meta.x0)
    g = similar(x, n)
    c = similar(x, m)
    v = fill!(similar(x, m), one(eltype(x)))
    Jtv = similar(x, n)
    return (
        nvar = n,
        ncon = m,
        objective = (@belapsed NLPModels.obj($nlp, $x)),
        gradient = (@belapsed NLPModels.grad!($nlp, $x, $g)),
        residual = (@belapsed NLPModels.cons!($nlp, $x, $c)),
        jtprod = (@belapsed NLPModels.jtprod!($nlp, $x, $v, $Jtv)),
    )
end

function print_timings(name, t)
    Printf.@printf(
        "%-10s nvar %6d ncon %6d | obj %.2e  grad %.2e  resid %.2e  jtprod %.2e\n",
        name, t.nvar, t.ncon, t.objective, t.gradient, t.residual, t.jtprod,
    )
    return
end

# Convenience: build + time each backend on one case.
function compare(name; backends = (:exa, :genopt, :arraydiff))
    ts = [opf_timings(opf_model(b, name)) for b in backends]
    for (t, b) in zip(ts, backends)
        print_timings(string(b), t)
    end
    return
end

case = "pglib_opf_case14_ieee.m"

exa = opf_model(:exa, case)
genopt = opf_model(:genopt, case)
arraydiff = opf_model(:arraydiff, case)
print_timings(case, opf_timings(exa))

case = "pglib_opf_case10000_goc.m"
exa = opf_model(:exa, case)
t = opf_timings(exa)
print_timings(case, t)
compare(case)
