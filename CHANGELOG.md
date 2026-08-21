# Changelog

## 0.1.0

Unreleased, and it runs. heddle is written in twill's `mode systems`, which
landed in twill 1.6; `twill test tests` passes all 8 suites under twill 1.7.1.
This entry used to say nothing here had executed, which is no longer true.
`README.md`'s State table says which piece each suite covers, and which two
pieces still have no test.

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

- A Laplace approximation: Newton to the mode, the Gaussian from `hessian`, the
  log evidence, sampling and the delta method.

Not written, and why: see the last section of `README.md`.

`docs/needs.md` is the list of language features this source asked twill for. It
now records which arrived and which are still open, and the open ones are 19,
22, 23, 25, 26 and 27.
