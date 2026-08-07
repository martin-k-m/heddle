# Changelog

## 0.1.0

Unreleased, and unrunnable. heddle is written in twill's `mode systems`, which
is still landing, so nothing here has executed.

Written:

- A model as `fn(Tensor) -> Tensor`, with the gradient of the log posterior
  supplied by twill's `grad`. No tracing layer and no second representation of
  the user's program.
- Distributions with log density, sampler, and the reparameterised form where
  one exists: normal, half-normal, lognormal, student t, exponential, gamma,
  beta, Dirichlet, categorical, multinomial, and a multivariate normal
  parameterised by its Cholesky factor.
- A differentiable `lgamma` over tensors, because a shape parameter is a
  parameter and `std/stats.tw`'s `F64` version is a constant to `grad`.
- Transforms for constrained parameters with their log Jacobians derived in the
  source: log, shifted log, logit, interval, stick-breaking simplex, ordered
  vectors, and a positive-diagonal Cholesky factor.
- Random walk Metropolis with a Robbins-Monro proposal scale.
- Static Hamiltonian Monte Carlo, as the reference the tree is checked against.
- NUTS with multinomial sampling, the generalised U-turn criterion and its two
  cross-boundary checks, dual averaging, and diagonal mass matrix adaptation on
  Stan's three-stage warmup schedule.
- Mean-field ADVI using the reparameterisation trick.
- Split rank-normalised R-hat, effective sample size by Geyer's initial
  monotone positive sequence, Monte Carlo standard error, divergence counts and
  tree depth saturation, with warnings written as sentences.
- Tests in the harness spool and loom share, and `examples/eight_schools.tw` in
  both parameterisations.

Not written, and why: see the last section of `README.md`.

The list of language features this source needs and twill does not have is
`docs/needs.md`. It is the useful output of this repository today.
