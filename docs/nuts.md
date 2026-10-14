# 3. Initialize, tune, and diagnose NUTS

Continue from [Write a NumPyro model](numpyro_model.md), using `model_normal`
with the original Gaussian likelihood and `sample_keys`, and the optimized
`popt` from the [Quickstart](quickstart.md). The goal is to obtain
reliable posterior samples, not simply to finish a run without an exception.

There are two numerical integrations involved. jnkepler integrates planetary
orbits using `dt`; NUTS integrates trajectories through **parameter space** using
its own step size. Adjusting one does not establish the accuracy of the other.
The NUTS mass matrix is unrelated to the planets' physical masses.

## Start near a good fit

Part 1 already optimized the orbital parameters and checked precision and
residuals. Reuse `popt`; there is no need to optimize it again here. This gives
the `fast` transit finder a starting point close to the observations. The
least-squares solution is an initial point, not a posterior sample or an
optimum of every possible likelihood. If you changed the forward model or
free parameters, update the fit and initialization accordingly.

NUTS usually explores correlated posteriors much more efficiently than random-walk MCMC.
If widely separated solutions are plausible, trying initial values near each
can help check posterior coverage.
In our experience with TTV analyses, however, NUTS still moves between modes more
readily than random-walk MCMC and can sometimes uncover additional solutions
even when transitions are rare. Uneven sampling of these modes often shows up
as elevated R-hat, low effective sample size, or differences between chain
traces, so pay attention to the [diagnostics below](#inspect-the-samples) in
such cases.

## Run with automatic adaptation

Use the four CPU devices configured in the [Quickstart session
setup](quickstart.md#set-up-the-session) to run `num_chains=4` chains in parallel,
as in the notebook.

```python
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_value

initial_values = {key: jnp.asarray(popt[key]) for key in sample_keys}
for key, value in initial_values.items():
    low, high = param_bounds[key]
    if not np.all((np.asarray(value) > low) & (np.asarray(value) < high)):
        raise ValueError(f"Initial {key} must lie strictly inside its prior bounds")

max_tree_depth = 8
kernel = NUTS(
    model_normal,
    init_strategy=init_to_value(values=initial_values),
    dense_mass=True,
    adapt_mass_matrix=True,
    regularize_mass_matrix=True,
    target_accept_prob=0.8,
    max_tree_depth=max_tree_depth,
)
mcmc = MCMC(
    kernel,
    num_warmup=1000,
    num_samples=1000,
    num_chains=num_chains,
    chain_method="parallel",
)
mcmc.run(
    random.PRNGKey(1), sample_keys, param_bounds,
    extra_fields=("num_steps", "accept_prob", "energy"),
)
mcmc.print_summary()
```

For the three-planet Kepler-51 TTV analysis in this tutorial, the code above
samples the posterior of the [Gaussian timing-error model](numpyro_model.md).
Sampling can take minutes or longer depending on the machine and sampler
initialization. For a **code-execution check only**, temporarily use
20 warmup / 20 sampling iterations. Such a run is too short to assess convergence
or obtain useful uncertainties; restore longer runs for inference.

The **setup & run HMC** section of the
[Kepler-51 notebook](https://github.com/kemasuda/jnkepler/blob/main/examples/kep51_ttv_normal.ipynb)
shows the complete example behind the runtime quoted in the Quickstart. Part 2
includes the same eccentricity-prior correction. The notebook also uses
Fisher-based initialization; here we start with automatic mass-matrix adaptation,
so the runtime need not match the notebook.

`init_to_value` takes values in the coordinates of the declared sample sites:
mass for `pmass`, log mass for `lnpmass`, and so on. Do not pass these physical
values as `MCMC.run(init_params=...)`, which expects internal unconstrained
coordinates. A bound-hitting optimum needs investigation before initialization.

Warmup adapts the sampler and is excluded from the retained draws. The settings
above are a starting configuration; use the diagnostics below to decide whether
to adjust them. Their meanings are explained under
[Understand the sampler settings](#understand-the-sampler-settings).

`chain_method="parallel"` runs one chain per JAX device. The CPU device count
must be configured before JAX initializes, as in part 1. The chains have
different random streams but all start at the same supplied optimum here.

## Inspect the samples

As in the Kepler-51 notebook, use ArviZ to inspect the sampler's step counts and
the traces of all parameters in `sample_keys`. The conversion preserves the
chain dimension for diagnostics. The plotting calls below use ArviZ 1.0 or
later; for older versions, replace them with the commented `az.plot_trace`
calls.

```python
import arviz as az

with numpyro.validation_enabled(False):
    idata = az.from_numpyro(mcmc, log_likelihood=True)

stats = mcmc.get_extra_fields(group_by_chain=True)
print("Divergences by chain:", np.asarray(stats["diverging"]).sum(axis=1))
print(
    "Fraction at the tree-depth cap:",
    float(np.mean(np.asarray(stats["num_steps"]) >= 2**max_tree_depth - 1)),
)

# Number of leapfrog steps per iteration (at most 255 for max_tree_depth=8).
az.plot_trace_dist(idata, group="sample_stats", var_names=["n_steps"], compact=False)
# ArviZ < 1.0:
# az.plot_trace(idata.sample_stats, var_names=["n_steps"], compact=False)
plt.show()

# Trace and marginal distribution of each inferred parameter.
az.plot_trace_dist(idata, var_names=sample_keys, compact=False)
# ArviZ < 1.0:
# az.plot_trace(idata, var_names=sample_keys, compact=False)
plt.tight_layout(pad=0.2)
plt.show()
```

`print_summary()` includes effective sample size (`n_eff`) and split R-hat
(`r_hat`); see the [NumPyro diagnostics reference](https://num.pyro.ai/en/stable/diagnostics.html).
R-hat near one indicates agreement of within-chain and between-chain variation;
ESS estimates how much independent information the correlated draws contain.
Use both alongside traces, divergences, and the precision needed for your result.
A large raw draw count is not a substitute for adequate ESS. A single chain
cannot provide a between-chain comparison.

Inspect every inferred parameter, not just mass ratios, and check derived
quantities central to your result. For fuller diagnostics, use ArviZ to examine
rank-normalized R-hat, bulk/tail ESS, and Monte Carlo errors. The example
notebooks show posterior export and plotting workflows. In the Kepler-51
notebook, see **plot models drawn from posterior** and **trace and corner
plots** to inspect uncertainties in both predictions and parameters.

## Understand the sampler settings

Use the diagnostics above to guide changes to the settings below. After
changing the kernel settings, construct a new `NUTS` kernel and `MCMC` object
and run warmup again. Compare ESS per unit time as well as divergences and
agreement between chains. The arguments are documented in
[NumPyro's NUTS API](https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc.NUTS).

### Step size: `target_accept_prob`

With step-size adaptation enabled (the default), `target_accept_prob` sets the
acceptance probability that warmup aims for when choosing the NUTS integrator's
step size. We start at `0.8`. Raising it to `0.9` or `0.95` usually produces
smaller steps, reducing integration error along the trajectory through
parameter space. This can help with divergences, at the cost of more steps to
travel a comparable distance.

Try a higher target if retained samples contain divergences, then check the
diagnostics again. Persistent divergences call for investigation of the model
and parameterization. If sampling is already well behaved, a higher target
does not necessarily give more useful samples per unit time.

### Trajectory length: `max_tree_depth`

NUTS grows a trajectory by doubling it until a stopping criterion is met or
`max_tree_depth` is reached. A depth of `8` allows at most `2**8 - 1 = 255`
leapfrog steps per iteration; depth `9` allows `511`, and depth `10` allows
`1023`. These are caps: an iteration can stop earlier.

If `n_steps` frequently reaches the cap, inspect parameter scaling and
correlations, and try increasing the depth by one. Keep the change if the
longer trajectories improve ESS per unit time. Each extra level roughly
doubles the maximum work per draw. Raising `target_accept_prob` can also make
the cap more relevant because smaller steps need more iterations of the
integrator to cover a similar distance. Increasing tree depth does not itself
reduce integration error or fix divergences.

### Parameter scales and correlations: the mass matrix

The mass matrix sets the scales and directions used to explore parameter
space. NumPyro works in its internal unconstrained coordinates here.

**`dense_mass`** chooses the matrix structure. With `False`, a diagonal matrix
accounts for each coordinate's scale. With `True`, a dense matrix can also
account for correlations between coordinates. We use `True` for the 15 orbital
parameters in this example. For larger models, dense blocks can be specified
with a list of tuples of sample-site names, as in the Student's t notebooks.

**`adapt_mass_matrix`** controls whether warmup estimates and updates the matrix
from the sampled positions. With `True`, no explicit initial matrix is needed;
NumPyro starts from the identity if none is supplied. With `False`, the matrix
stays fixed at its supplied or default value. Adaptation ends after warmup,
so increasing `num_samples` does not give it more time to learn the geometry;
increase `num_warmup` if that is needed.

The Kepler-51 notebook supplies a Fisher-based initial estimate through
`inverse_mass_matrix`. With adaptation enabled, warmup can refine this
estimate. See [Fisher information and mass-matrix initialization](mass_matrix.md)
for the matrix convention, coordinate transformations, and code. In particular,
this argument takes an estimate of the **parameter covariance** in
unconstrained coordinates, obtained by inverting the Fisher matrix.

**`regularize_mass_matrix=True`** stabilizes the covariance estimates made
during warmup. It only applies when mass-matrix adaptation is enabled. Keep
adaptation and regularization enabled initially; notebook settings that disable
them are choices for particular analyses.

## Tune in response to a problem

| What you see | What to investigate or change |
| --- | --- |
| Failure to initialize, or non-finite values | Check site names, prior bounds, units, eccentricities, transit matching, and model values/gradients near the initial solution. More samples will not repair the model. |
| Divergences in retained samples | Inspect where they occur. Try a higher `target_accept_prob` such as 0.9, then reassess; persistent problems call for better parameterization, initialization, or model checks. |
| Frequent tree-depth saturation | Check correlations and scaling. Increase `max_tree_depth` if longer trajectories improve ESS per unit time; each extra level can roughly double the maximum work per draw. |
| Poor mixing or disagreement across chains | Revisit warmup length, dense adaptation, parameterization, and alternate modes. Longer runs help only if chains can explore the relevant posterior. |
| Good sampler diagnostics but structured timing residuals | Revisit the likelihood, dynamics, observations, and numerical accuracy. Sampler tuning cannot repair a misspecified model. |

Continue to [Models and extensions](models.md) for other observables, or choose
a complete analysis from
[the example notebooks](examples.md).
