# Advanced: Fisher information and the NUTS mass matrix

This is optional material after [the NUTS tutorial](nuts.md). Start with automatic
adaptation first. The mass matrix describes scales and correlations for motion
through parameter space; it has no connection to stellar or planetary masses.
The **setup & run HMC** section of the
[Kepler-51 notebook](https://github.com/kemasuda/jnkepler/blob/main/examples/kep51_ttv_normal.ipynb)
shows a complete use of Fisher-based initialization. This appendix explains
that code using the notebook's variable names and sampler settings.

## Which matrix goes where?

Let $z$ denote NumPyro's **unconstrained** parameter coordinates. A bounded
sample site is transformed internally: for example, a Uniform variable between
$a$ and $b$ uses a logit coordinate $z=\log[(x-a)/(b-x)]$.
Even `lnpmass` needs a further transformation if it has bounded support.

For fixed, independent Gaussian timing uncertainties, define standardized
residuals $r_i(z)=[t_i^{\mathrm{obs}}-t_i(z)]/\sigma_i$ and their Jacobian
$J_{ij}=\partial r_i/\partial z_j$. The jnkepler helper computes
$F=J^\mathsf{T}J$. Its inverse approximates the local parameter covariance
$C_z\approx F^{-1}$ when the linearized likelihood is informative and dominates
the local geometry.

| Quantity | Meaning | Use |
| --- | --- | --- |
| `info["fisher"]` | $F$, local likelihood information in the requested coordinates | Inspect before inversion |
| `parameter_cov_unconstrained` | $F^{-1}$, approximate covariance of $z$ | Initial guess for NumPyro's `inverse_mass_matrix` |
| NUTS mass matrix $M$ | Momentum covariance in $K(p)=p^\mathsf{T}M^{-1}p/2$ | Its inverse is the API argument above |

Thus the object named **`inverse_mass_matrix` takes the parameter covariance
estimate**, not the Fisher matrix itself. NumPyro's warmup implementation also
uses an estimated parameter covariance for this quantity; see its
[mass-matrix adaptation source](https://num.pyro.ai/en/stable/_modules/numpyro/infer/hmc_util.html#warmup_adapter).

This is a local likelihood approximation, not the full posterior Hessian or a
posterior uncertainty estimate. The helper does not add prior curvature or the
curvature of the coordinate-transform log density. Nonlinear degeneracies and
multiple modes cannot be captured by one fixed matrix.

## Compute the Fisher matrix at the Kepler-51 fit

Start after the notebook's optimization and model-definition cells. They
provide `jttv`, `param_bounds`, the least-squares solution `popt`, and the
Gaussian timing model `model_indep_normal`. Keep the session setup, including
64-bit JAX and `num_chains`, from the notebook or [Quickstart](quickstart.md).
If continuing from the tutorial instead, first set
`model_indep_normal = model_normal`; the model on the preceding pages uses
that shorter name. The code below uses the notebook's `sample_keys` order.

```python
from jnkepler.information import information_from_model_independent_normal

sample_keys = ["ecosw", "esinw", "pmass", "period", "tic"]

info = information_from_model_independent_normal(
    model=model_indep_normal,
    model_args=(sample_keys, param_bounds),
    keys=sample_keys,
    param_space="unconstrained",
    pdic=popt,
    mu_name="tcmodel",
    observed=jttv.tcobs_flatten,
    sigma_sd=jttv.errorobs_flatten,
)
```

This evaluates how the predicted transit times change when the orbital
parameters move slightly away from `popt`. `model_args` supplies the arguments
to `model_indep_normal`, and `mu_name="tcmodel"` selects the predicted times
recorded by `numpyro.deterministic("tcmodel", tcmodel)` inside that model.
`observed` and `sigma_sd` supply the measured times and their fixed standard
deviations. The helper differentiates the standardized residuals and returns
`info["fisher"]`, the $J^\mathsf{T}J$ matrix defined above. It does not run
another optimization or sample the posterior.

`pdic=popt` supplies values in the **declared sample-site coordinates**: for
example, `popt["pmass"]` contains mass ratios, not their logits.
`param_space="unconstrained"` asks the helper to transform those values using
the model's prior supports before differentiating. The resulting matrix is
therefore in the coordinates that NUTS actually uses. Changing prior bounds
changes this transformation, so recompute the matrix if the bounds change.

## Convert it to a dense orbital block

The next three lines follow the notebook directly:

```python
dense_mass = [tuple(sample_keys)]
parameter_cov_unconstrained = jnp.linalg.inv(info["fisher"])
inverse_mass_matrix = {dense_mass[0]: parameter_cov_unconstrained}
```

`dense_mass` groups all five orbital sample sites into one block. Each site
contains three planet values in this example, so the block is a **15 by 15**
matrix. It can describe correlations between different parameters and between
different planets, rather than assigning an independent scale to each value.

`jnp.linalg.inv` converts the local information matrix into the covariance
approximation $C_z$. Its diagonal describes approximate squared widths in
unconstrained coordinates; its off-diagonal entries describe covariances.
Supplying this matrix gives NUTS an initial estimate of the scales and
correlations near the fitted solution.

`inverse_mass_matrix` is a dictionary whose key identifies the sample sites
covered by that matrix. Here `dense_mass[0]` is
`("ecosw", "esinw", "pmass", "period", "tic")`, so NumPyro knows how to map
the matrix entries to the sampled parameters.

The helper concatenates columns in `keys` order, flattening each site's array.
For this example, the order starts with `ecosw[0:3]`, then `esinw[0:3]`,
`pmass[0:3]`, `period[0:3]`, and `tic[0:3]`. Use **exactly that same order**
in the tuple key of `inverse_mass_matrix` and the tuple in `dense_mass`.
`info["col_names"]` and `info["col_slices"]` expose the mapping.
Deterministic sites such as `ecc` and `tcmodel` are not sampled coordinates
and must not be included.

## Use the matrix to start NUTS

Pass the block structure and its covariance estimate to the kernel, then run
warmup and sampling as in the notebook:

```python
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value

kernel = NUTS(
    model_indep_normal,
    init_strategy=init_to_value(values=popt),
    dense_mass=dense_mass,
    inverse_mass_matrix=inverse_mass_matrix,
    max_tree_depth=8,
    target_accept_prob=0.8,
    regularize_mass_matrix=False,
    adapt_mass_matrix=True,
)

mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)
rng_key = random.PRNGKey(0)
mcmc.run(
    rng_key, sample_keys, param_bounds,
    extra_fields=(
        "potential_energy", "num_steps", "adapt_state", "energy",
        "accept_prob", "mean_accept_prob",
    ),
)
```

`init_to_value(values=popt)` chooses the starting **position**, while
`inverse_mass_matrix` sets the initial scales and correlations for motion
around that position. Both use the same fitted solution here, but serve
different purposes. The posterior being sampled is still defined by
`model_indep_normal`, including its eccentricity-prior factor; the Fisher
approximation only initializes the sampler.

With `adapt_mass_matrix=True`, warmup estimates a covariance from the sampled
positions and updates the supplied matrix. Thus the Fisher estimate helps
initialize exploration, but it is not held fixed throughout this run.
Setting `adapt_mass_matrix=False` would keep it fixed, while still allowing
step-size adaptation by default.

`regularize_mass_matrix=False` reproduces the Kepler-51 notebook setting:
the covariance estimates made during warmup are not regularized. The
[introductory NUTS example](nuts.md#run-with-automatic-adaptation) uses `True`
to stabilize those estimates. This option acts on warmup estimates; it does
not regularize the initial `jnp.linalg.inv(info["fisher"])` above.

The run uses 1000 warmup iterations followed by 1000 retained draws per chain.
The extra fields record sampler diagnostics. Use the [NUTS diagnostic
plots](nuts.md#inspect-the-samples) to inspect the result and compare ESS per
unit time with automatic initialization.

## Check the approximation when adapting the example

This estimate is local to `popt`. It can work very well when the posterior is
approximately multivariate normal in unconstrained coordinates and the
likelihood dominates its curvature: the local scales and correlations then
represent the posterior globally. Sampling can struggle when those scales or
correlation directions change substantially across the posterior, as in curved
or funnel-shaped distributions. Warmup adaptation can improve the initial
estimate, but a single fixed matrix cannot follow such changes with position.

The direct inversion above assumes a finite, positive-definite, sufficiently
well-conditioned Fisher matrix. A very large condition number can make the
inverse unreliable even if all eigenvalues are positive. If this fails for
your model, inspect weakly constrained directions, the parameterization, and
the fitted point. A pseudoinverse with zero eigenvalues is not a valid
positive-definite metric. To use automatic initialization instead, omit
`inverse_mass_matrix` from `NUTS` and keep `adapt_mass_matrix=True`.

## Why the Student's t notebooks use separate blocks

The Student's t examples use one dense orbital block and another for `lndf`
and `lnvar`. Their Gaussian information helper is applied only to the timing
prediction with fixed observational errors. This gives a **Gaussian surrogate
for orbital geometry**, not the Fisher information of the Student's t model.
It does not estimate the noise block or orbital-noise cross-correlations.

The supplied noise-block variances must be in the unconstrained coordinates of
those bounded sample sites too. Do not copy a physical-space covariance, a
different planet count, or different prior bounds into this initialization.
When you change the model, recompute the approximation or use automatic dense
adaptation. Freezing an inaccurate matrix with `adapt_mass_matrix=False` can
make sampling substantially harder.
