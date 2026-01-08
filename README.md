# ArrayDiff

| **Build Status** |
|:----------------:|
| [![Build Status][build-img]][build-url] [![Codecov branch][codecov-img]][codecov-url] |

Experimental addition of array support to `MOI.Nonlinear.ReverseAD`

> [!WARNING]
> This code is still very much experimental

You need to use the following branch of [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl)
```julia
Pkg.add(PackageSpec(name="MathOptInterface", rev="bl/arraydiff"))
```

Supported operators:

- [x] `vect`, e.g, `[1, 2]`.
- [x] `dot`
- [x] `row`, e.g. `[1 2; 3 4]`
- [x] `hcat`
- [x] `vcat`
- [x] `norm`
- [x] Matrix-Vector product
- [x] Matrix-Matrix product
- [ ] Broadcasting scalar operator

Supported levels of AD:

- [x] 0-th order
- [x] 1-st order
- [ ] 2-nd order

[build-img]: https://github.com/blegat/ArrayDiff.jl/actions/workflows/ci.yml/badge.svg?branch=main
[build-url]: https://github.com/blegat/ArrayDiff.jl/actions?query=workflow%3ACI
[codecov-img]: https://codecov.io/gh/blegat/ArrayDiff.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/blegat/ArrayDiff.jl/branch/main
