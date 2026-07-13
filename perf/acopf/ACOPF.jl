# Vectorized AC-OPF on ArrayDiff — entry point. `include` this file.
#
# * `structured.jl` — GPU-friendly constant matrix types (gather/scatter and
#   ELLPACK) that ArrayDiff keeps by reference on the tape.
# * `data.jl`      — case9mod (JuMP tutorial) and PowerModels-derived data.
# * `solver.jl`    — first-order augmented-Lagrangian solver (projected Adam),
#   built on `eval_residual!` / `eval_residual_jtprod!`; storage-generic.
# * `models.jl`    — the two model builders: `build_rect` (JuMP-tutorial form,
#   rectangular voltages + Ybus) and `build_polar` (GenOpt/ExaModels form,
#   polar voltages with sin/cos branch flows).

import ArrayDiff
import JuMP
import LinearAlgebra
import MathOptInterface as MOI
import SparseArrays

include("structured.jl")
include("data.jl")
include("solver.jl")
include("models.jl")
