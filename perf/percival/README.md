# AC-OPF through NLPModels solvers (Percival / MadNLP) with ArrayDiff

Solve the two vectorized AC-OPF forms from `../acopf` through the **NLPModels**
interface, so any NLPModels solver can consume them, with ArrayDiff providing
the (first-order, vectorized, GPU-ready) derivatives.

```
JuMP model (ArrayOfVariables + array expressions)
      │  build objective + constraint-group residual evaluators (ArrayDiff)
      ▼
ArrayDiffNLPModel <: NLPModels.AbstractNLPModel      (adnlp.jl)
   obj/grad         ← objective evaluator (reverse mode)
   cons/jprod/jtprod ← vectorized residual evaluators (one fused pass each)
   jac_coord        ← Jacobian materialized from jtprod (for interior-point)
      ▼
NLPModels solver:  MadNLP (CompactLBFGS) │ Percival (AL)
```

`min f(x)  s.t.  c(x) = 0,  l ≤ x ≤ u`. Inequalities (thermal limits, voltage
magnitude) become equalities with box-constrained slacks, so only equality
constraints + variable bounds remain.

## Results (CPU, case objectives vs Ipopt)

| case | solver | gap | feasibility | status | iters |
|---|---|---|---|---|---|
| rect case9mod | MadNLP/LBFGS | 0.0000% | 1.5e-15 | SOLVE_SUCCEEDED | 27 |
| rect case9mod | Percival/LBFGS+Tron | 0.0000% | 1.9e-10 | max_iter | 301 |
| polar case9 | MadNLP/LBFGS | 0.0000% | 1.4e-14 | SOLVE_SUCCEEDED | 31 |
| polar case9 | Percival/LBFGS+Tron | -0.008% | 5.4e-05 | stalled | 30 |
| polar case14 | MadNLP/LBFGS | 0.0000% | 4.9e-15 | SOLVE_SUCCEEDED | 42 |
| polar case14 | Percival/LBFGS+Tron | -2.5% | 6.8e-03 | max_time | 10 |
| polar case30 | MadNLP/LBFGS | 0.0000% | 1.7e-14 | SOLVE_SUCCEEDED | 31 |
| polar case30 | Percival/LBFGS+Tron | -43% | 8.0e-02 | stalled | 24 |

**MadNLP in quasi-Newton mode (`hessian_approximation = CompactLBFGS`) is the
robust winner**: it converges tightly and reliably on every case using only
gradients + the constraint Jacobian (no exact Hessian). Percival (augmented
Lagrangian) is correct on the small cases but its first-order subproblem
(LBFGS-approximated AL Hessian) fails to certify KKT as the penalty grows and
degrades on larger cases within the iteration/time budget.

## Files

| file | purpose |
|---|---|
| `adnlp.jl` | `ArrayDiffNLPModel` — the NLPModels bridge (obj/grad/cons/jprod/jtprod/jac_coord) |
| `build_percival.jl` | build rect / polar AC-OPF as `ArrayDiffNLPModel`s (reuses `../acopf`) |
| `run_madnlp.jl` | MadNLP quasi-Newton driver |
| `run_percival.jl` | Percival driver (LBFGS subproblem modifier) |
| `gpu_percival.jl` | JLArray (GPU-semantics) driver + CPU-vs-device eval checks |
| `compare.jl` | side-by-side MadNLP vs Percival on all cases |

## Percival GPU-compatibility fixes (in `~/.julia/dev/Percival`)

Percival could not run on GPU-resident (`CuVector` / `JLArray`) models. Fixes:

1. **`AugLagModel.store_Jv/store_Jtv`** were hardcoded `Vector{T}`; used in
   `grad!` (`g .+= jtprod!(model, x, μc_y, store_Jtv)`) they forced a device
   mismatch. Changed to the storage type `V`.
2. **`AugLagModel`'s meta** ran `findall`-based variable-bound analysis, which
   scalar-indexes GPU arrays. Disabled (`variable_bounds_analysis = false`);
   the subproblem solver only projects with `lvar/uvar`.
3. **Dispatch guards** (`percival(nlp)`, `percival(Val{:equ})`) called
   `NLPModels.equality_constrained`, whose fallback scalar-iterates the bound
   vectors. Added `_is_equality_constrained` which compares them on the host.
4. **`SPGSubSolver`** (`src/spg_subsolver.jl`): a spectral projected gradient
   subproblem solver that is pure broadcasts — a GPU-compatible replacement for
   the default `TronSolver`, whose projected-CG inner loop uses scalar `x[i]`
   indexing that GPU arrays disallow. Selected with `subsolver = SPGSubSolver`;
   being first-order it needs no quasi-Newton wrapper.

With these, `percival(nlp; subsolver = SPGSubSolver)` runs the **entire AL loop
on `JLArray` storage with scalar indexing disallowed** — validated end-to-end
(no scalar-indexing errors). Convergence of the pure-first-order SPG path on
ill-conditioned AC-OPF is slow (as expected); the fixes make Percival
*GPU-runnable*, and for GPU production MadNLPGPU (below) is the stronger option.

Tests: `Percival/test/spg_subsolver.jl` (in the Percival test suite).

## MadNLP on GPU

MadNLP consumes the same `ArrayDiffNLPModel` and has first-class GPU support via
**MadNLPGPU** (cuDSS/cuSOLVER KKT solves). It was not runnable in this
container (no CUDA; JLArray has no dense factorization backend), but the model
is device-generic, so on real hardware `MadNLPGPU.CUDSSSolver` + a CuArray tape
is the intended GPU path. `jac_coord!` currently materializes a dense Jacobian
(fine for these sizes); large GPU problems want a sparse/matrix-free assembly.

## Remaining work: full `JuMP → MOI → NLPModelsJuMP.Optimizer` routing

`ArrayDiffNLPModel` is built directly from ArrayDiff evaluators here. To go
through `NLPModelsJuMP.Optimizer` end-to-end (as the NLS path already does),
vector nonlinear constraints must survive `JuMP.@constraint`: today
`@constraint(m, arr_expr in MOI.Zeros(n))` fails because JuMP scalar-indexes the
`GenericArrayExpr` (which ArrayDiff blocks). The steps are:

1. ArrayDiff JuMP layer: `JuMP.build_constraint(err, ::GenericArrayExpr, ::VLS)`
   + a vector shape so the constraint is kept whole (no scalarization).
2. NLPModelsJuMP ArrayDiff ext: collect `ArrayNonlinearFunction`-in-`Zeros`
   constraints and build `ArrayDiffNLPModel` (move the type here from `adnlp.jl`).
3. Route `Optimizer.copy_to` to that builder when the backend is `ArrayDiff.Mode`
   and vector constraints are present.
