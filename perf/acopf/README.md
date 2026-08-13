# Vectorized AC-OPF on ArrayDiff (first-order, GPU-ready)

Two AC-OPF formulations written as *whole-vector* expressions
(matrix–vector products + broadcasts) so ArrayDiff's array tape evaluates
them with a handful of `mul!`/broadcast kernels — the same code path on CPU
(`Vector{Float64}`), JLArrays (GPU semantics without a GPU) and CUDA
(`CuVector{Float64}`):

1. **rect** — the JuMP tutorial (`optimal_power_flow.jl`) form: rectangular
   complex voltages, power balance `S_G - S_D = V .* conj(Y V)` split into
   real/imaginary parts with the sparse bus admittance components `G`, `B`.
2. **polar** — the GenOpt/ExaModels (`examples/opf.jl`) form: polar voltages,
   branch flows with `sin`/`cos` of angle differences, incidence
   gather/scatter for the nodal balance.

Since AC-OPF needs second-order info for interior-point methods but ArrayDiff
is first-order (by design, for now), the problems are solved with a
first-order augmented Lagrangian (`solver.jl`): equality residual groups
compiled to `eval_residual!`/`eval_residual_jtprod!`, inequalities converted
to equalities with box slacks, variable bounds by projection, inner loop =
projected Adam. Don't expect Ipopt-grade precision — the point is the
vectorized evaluation pipeline, not the outer optimizer.

## Structured constants (new ArrayDiff feature exercised here)

Constant arrays that are *not* dense `Array`s are now kept **by reference**
on the tape (`Expression.arrays`, `NODE_ARRAY_VALUE` leaf) instead of being
serialized; matmul nodes call `LinearAlgebra.mul!` directly on them.
`structured.jl` provides two purpose-built types whose products are pure
broadcasts (GPU-safe, no atomics, no CUSPARSE dependency):

* `GatherMatrix` — one 1 per row: `A*x` is a gather, `A'*w` a padded
  fixed-width gather-accumulate (max-degree many fused broadcasts; no
  atomics, no scan). Encodes branch↔bus / gen↔bus incidence.
* `ELLMatrix` — padded fixed-width sparse rows (ELLPACK), transpose stored
  explicitly. Encodes the admittance components `G`, `B`; better suited to
  GPUs than `SparseMatrixCSC` for the short uniform rows of power networks.

`SparseMatrixCSC` also works (it just goes through its own `mul!`).

## Files

| file | purpose |
|---|---|
| `structured.jl` | `GatherMatrix`, `ELLMatrix`, `map_storage` (device transfer) |
| `data.jl` | `case9mod()` (tutorial data, per-unit) and `parse_polar_case` (PowerModels) |
| `models.jl` | `build_rect`, `build_polar` → `ALProblem` |
| `solver.jl` | first-order AL (projected Adam), storage-generic |
| `reference.jl` | Ipopt references (scalar JuMP model / PowerModels) |
| `main.jl` | CPU end-to-end: solve both forms, compare with Ipopt |
| `check_derivatives.jl` | finite-difference checks of all residual groups |
| `gpu_check.jl` | run everything on `JLArray` with scalar indexing disallowed |
| `run_gpu.jl` | CUDA driver (needs a GPU machine; `Pkg.add("CUDA")` first) |

## Running

```sh
julia --project=. check_derivatives.jl   # derivative correctness
julia --project=. main.jl                # CPU solves vs Ipopt
julia --project=. gpu_check.jl           # GPU-semantics via JLArrays
julia --project=. run_gpu.jl case9.m     # on a machine with an NVIDIA GPU
```

The development container had no GPU: `run_gpu.jl` is untested on real
hardware, but `gpu_check.jl` runs the identical code paths under GPUArrays'
scalar-indexing ban, which catches the class of bugs that breaks CUDA runs
(it already caught one: `cumsum!` on vectors silently falls back to a scalar
loop on GPU arrays).

## Results (CPU, this container)

Objectives vs Ipopt with the default Adam + SPG-polish schedule:

| problem | Ipopt | first-order AL | gap | max violation |
|---|---|---|---|---|
| rect case9mod | 3087.84 | 3088.09 | 0.008% | 3.5e-7 |
| polar case9 | 347.70 | 348.39 | 0.20% | 1.5e-7 |
| polar case14 | 8081.52 | 8092.90 | 0.14% | 1.8e-7 |
| polar case30 | 204.97 | 205.89 | 0.45% | 7.5e-7 |

`bench_cpu.jl` (AL gradient = 8 residual forward+J'v passes + objective
gradient, polar form):

| case | GatherMatrix | SparseMatrixCSC |
|---|---|---|
| case9 (9 buses) | 64 µs | 52 µs |
| case118 (118 buses) | 281 µs | 110 µs |
| case1354 (1354 buses) | 2.75 ms | 1.54 ms |

On CPU, `SparseMatrixCSC`'s tight loops win — use `use_gather = false`
there. The structured types exist for the GPU, where kernel count, coalesced
access and the absence of row-pointer indirection/atomics matter;
`run_gpu.jl` times GatherMatrix against CUSPARSE CSR to check that claim on
real hardware.
