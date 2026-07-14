# ArrayDiff

| **Build Status** |
|:----------------:|
| [![Build Status][build-img]][build-url] [![Codecov branch][codecov-img]][codecov-url] |

Experimental addition of array support to `MOI.Nonlinear.ReverseAD`

> [!WARNING]
> This code is still very much experimental. First-order is mostly working but second-order isn't implemented yet.

## Presentations

* Adding array support for JuMP’s Automatic Differentiation at JuMP-dev 2026 by Sophie Lequeu [[slides](https://jump.dev/assets/jump-dev-workshops/2026/slides_sophie.pdf)] [[video](Adding array support for JuMP’s Automatic Differentiation)]
* Experiments with Vector-Valued Nonlinear Functions in JuMP by Benoît Legat [[slides](https://jump.dev/assets/jump-dev-workshops/2026/slides_siam_benoit.pdf)]
* Accelerating energy system optimization on GPUs by Benoît Legat [[slides](https://blegat.github.io/slides/2026_IFORS)]

[build-img]: https://github.com/blegat/ArrayDiff.jl/actions/workflows/ci.yml/badge.svg?branch=main
[build-url]: https://github.com/blegat/ArrayDiff.jl/actions?query=workflow%3ACI
[codecov-img]: https://codecov.io/gh/blegat/ArrayDiff.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/blegat/ArrayDiff.jl/branch/main
