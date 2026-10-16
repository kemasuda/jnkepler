# 2. Write a NumPyro model

Continue in the session from the [Quickstart](quickstart.md), keeping the
optimized parameters `popt` and search bounds `param_bounds`. We now
describe which parameters are uncertain and how the measured timings relate
to a jnkepler prediction. NumPyro defines the probability model; jnkepler
supplies the differentiable N-body prediction inside it.

We fix the stellar mass, assume coplanar edge-on orbits, and use independent
Gaussian timing errors.

Separately, we choose priors on the orbital parameters. The model below uses
uniform priors on planet-to-star mass ratios, periods, and conjunction epochs.
For eccentricity, we sample bounded `ecosw` and `esinw` and apply the Kepler-51
example's `eprior` correction to obtain an approximately uniform prior on $e$.
The [prior discussion below](#understand-the-prior-before-changing-coordinates)
explains this correction and the effect of the bounds.

## Choose parameters and bounds

```python
import numpyro
import numpyro.distributions as dist

sample_keys = ["ecosw", "esinw", "period", "pmass", "tic"]
```

We reuse the bounds from `ttv_default_parameter_bounds(jttv)` in part 1.
The helper supplies lower and upper arrays for each parameter. In this example,
the period bounds are within 1% of `p_init`, conjunction epochs within 0.2 days
of each planet's first observation, mass ratios between `1e-7` and `1e-3`, and
each eccentricity component between -0.2 and 0.2. Inspect and change these bounds
for your own system. They become priors only because we use them in distributions
below; the helper itself merely constructs bounds.

`smass`, `cosi`, and `lnode` are fixed here. A parameter is inferred only if it
has an unobserved `numpyro.sample` site. You can fix a parameter by putting its
value directly in `par`, but also update the optimization and initialization
steps to match the set of free parameters.

## Define the model

```python
def model_normal(sample_keys, param_bounds):
    par = {"smass": 1.0}

    # Bounded distributions: one value per planet at each sample site.
    for key in sample_keys:
        low, high = param_bounds[key]
        par[key] = numpyro.sample(
            key, dist.Uniform(jnp.asarray(low), jnp.asarray(high))
        )

    # Also support sampling log mass instead of mass in an alternative model.
    if "lnpmass" in sample_keys:
        par["pmass"] = numpyro.deterministic("pmass", jnp.exp(par["lnpmass"]))

    ecc = numpyro.deterministic("ecc", jnp.hypot(par["ecosw"], par["esinw"]))
    # Correct the component prior toward a uniform eccentricity prior.
    eps_ecc = 1e-4
    numpyro.factor("eprior", -jnp.log(ecc + eps_ecc))

    tcmodel, ediff = jttv.get_transit_times_obs(par)
    numpyro.deterministic("tcmodel", tcmodel)
    numpyro.deterministic("ediff", ediff)

    # Likelihood: the measured timings, in days and grouped by planet.
    numpyro.sample(
        "obs",
        dist.Normal(tcmodel, jnp.asarray(jttv.errorobs_flatten)),
        obs=jnp.asarray(jttv.tcobs_flatten),
    )
```

The NumPyro operations used here are:

| Operation | Role in this analysis |
| --- | --- |
| `sample(name, distribution)` | Declare a parameter and its prior |
| `factor(name, log_factor)` | Add a log-density term, here to modify the eccentricity prior |
| `deterministic(name, value)` | Record a derived value, without adding a prior |
| `sample(..., obs=data)` | Add the likelihood of the fixed observations |

The model is run under NumPyro's inference machinery; it is not a function that
returns fitted parameters. As in the notebook, the sample sites use arrays of
independent distributions. NumPyro sums their elementwise log densities and
the `eprior` terms when evaluating the joint log density for NUTS. See
[NumPyro's primitives](https://num.pyro.ai/en/stable/primitives.html) for the
general API. The effective prior includes the `eprior` factor as well as the
sample-site distributions; it is not uniform in the eccentricity components.

Inside the model, use `jax.numpy` for calculations involving sampled values.
Load files and prepare observation arrays outside it. Do not convert a sampled
value to a Python `float` or NumPy array: NUTS needs JAX to differentiate through
the computation.

## Understand the prior before changing coordinates

Uniform priors depend on the coordinates in which they are defined.

**Mass versus log mass.** The model above is uniform in mass ratio. To instead
use a log-uniform prior, replace `pmass` in `sample_keys` with `lnpmass` and use
the helper's logarithmic bounds. The conversion to `pmass` is already in the
model. Sample only one of these two sites. Uniform `lnpmass` implies a density
proportional to $1/\mathrm{pmass}$ within the specified mass limits; it changes
the prior, not just the numerical scaling. Update the initial values and any
mass-matrix blocks when you change site names.

**Eccentricity components versus eccentricity.** Independent uniform
$h=e\cos\omega$ and $k=e\sin\omega$ on a square do not give a uniform prior on
$e$. In the part of the disk lying inside that square, the area element is
$dh\,dk=e\,de\,d\omega$, so the marginal density grows with $e$.
Also, the helper's `emax` bounds each component: it is not a radial cutoff
$e<\mathrm{emax}$.

The `eprior` factor in our model follows **setup & run HMC** in the
[Kepler-51 notebook](https://github.com/kemasuda/jnkepler/blob/main/examples/kep51_ttv_normal.ipynb).
An exact correction of $-\sum\log e$ to the log density would multiply the
component-space density by $\prod 1/e$, cancelling the Jacobian factor within
the allowed region. This changes the prior; merely recording `ecc` with
`numpyro.deterministic` would not do so.

Following the notebook, we use `eps_ecc=1e-4` in each planet's
`-log(ecc + eps_ecc)` term to soften the log singularity at zero. The radial
density is then proportional to $e/(e+\mathrm{eps\_ecc})$ where all angles are
allowed: approximately flat away from zero. The square component bounds also
restrict the angular range at larger $e$. Thus the resulting prior is only
approximately uniform in eccentricity, not exactly uniform on a specified
eccentricity disk.

NumPyro handles the transformations needed for its internal unconstrained
sampling coordinates. That automatic transformation does not change a prior
you wrote on mass into a uniform prior on log mass, or choose an eccentricity
prior for you.

## Change the noise model

These are **replacement likelihood blocks inside the model**, after computing
`tcmodel`. Use one likelihood named `obs`, and keep `model_normal` unchanged if
continuing with the Gaussian walkthrough on the next page.

### Gaussian noise with additional scatter

For a single extra timing-scatter parameter shared by all observations:

```python
# Replace the likelihood block inside a copy of model_normal.
jitter = numpyro.sample("jitter", dist.HalfNormal(1.0 / 1440.0))
sigma = jnp.sqrt(jnp.asarray(jttv.errorobs_flatten)**2 + jitter**2)
numpyro.sample(
    "obs", dist.Normal(tcmodel, sigma),
    obs=jnp.asarray(jttv.tcobs_flatten),
)
```

Here the illustrative prior scale is one minute, expressed in days. Choose it
for the measurement process. The predicted `sigma` must appear in the full
distribution, including its normalization; minimizing residuals divided by a
free `sigma` alone does not define the same likelihood.

### Student's t noise

For heavier tails and a fitted common scale multiplier:

```python
# Replace the likelihood block inside another copy of model_normal.
lndf = numpyro.sample("lndf", dist.Uniform(jnp.log(0.5), jnp.log(100.0)))
lnvar = numpyro.sample("lnvar", dist.Uniform(jnp.log(0.1), jnp.log(10.0)))
df = numpyro.deterministic("df", jnp.exp(lndf))
scale_multiplier = numpyro.deterministic("scale_multiplier", jnp.exp(0.5 * lnvar))
numpyro.sample(
    "obs",
    dist.StudentT(
        df=df, loc=tcmodel,
        scale=jnp.asarray(jttv.errorobs_flatten) * scale_multiplier,
    ),
    obs=jnp.asarray(jttv.tcobs_flatten),
)
```

The bounds follow the
[V1298 Tau notebook](https://github.com/kemasuda/jnkepler/blob/main/examples/v1298tau_ttv_student.ipynb)
(**setup & run HMC**) and the
[TRAPPIST-1 notebook](https://github.com/kemasuda/jnkepler/blob/main/examples/trappist-1_ttv_student.ipynb)
(**Running NUTS**), and should be revisited for a new
analysis. Despite the historical site name `lnvar`, `exp(lnvar)` is the squared
scale multiplier, **not the variance** of the Student's t distribution.
For `df > 2`, its standard deviation is `scale * sqrt(df / (df - 2))`;
for `df <= 2`, no finite variance exists. See the
[StudentT reference](https://num.pyro.ai/en/stable/distributions.html#numpyro.distributions.continuous.StudentT).

The notebooks express this likelihood using residuals divided by fixed
observational errors. With those errors fixed, the timing-space form above
differs only by a parameter-independent normalization and gives the same
posterior. If you divide residuals by a *fitted* scale, the normalization is
parameter-dependent and cannot be dropped.

When extending the model, initialize the new noise sites as well as the orbital
sites, and reassess diagnostics. The Gaussian least-squares optimum is only a
starting point for a Student's t posterior.

## RV likelihoods and joint inference

The rest of this page covers optional extensions. Continue directly to
[NUTS](nuts.md) for the Gaussian TTV walkthrough. The
[Models and extensions guide](models.md#radial-velocities) explains how to
obtain `rv_model` from orbital parameters; the block below describes the
measurement model for a single instrument. RV predictions, data, uncertainties,
offsets, and jitter are all in **m/s**.

Add this block **inside your NumPyro model**, after computing `rv_model` from
its sampled orbital parameters. Prepare the observed velocities `rv_obs` and
uncertainties `rv_error` outside the model, in the same order as `times_rv`.

```python
# Inside the NumPyro model, after computing rv_model.
gamma_rv = numpyro.sample("gamma_rv", dist.Normal(0.0, 100.0))
jitter_rv = numpyro.sample("jitter_rv", dist.HalfNormal(5.0))
sigma_rv = jnp.sqrt(jnp.asarray(rv_error)**2 + jitter_rv**2)
numpyro.sample(
    "rv_obs",
    dist.Normal(rv_model + gamma_rv, sigma_rv),
    obs=jnp.asarray(rv_obs),
)
```

The offset and jitter priors are illustrative; adapt them to your data's zero point and
instrument. With multiple instruments, give each an offset and jitter, indexed
by the instrument of each observation.

For joint inference, replace the original `get_transit_times_obs` call with
`tcmodel, rv_model, ediff = jttv.get_transit_times_and_rvs_obs(par, times_rv)`.
Retain the timing likelihood and add the RV block above. The two observed sites add
their log likelihoods, assuming independent measurement errors conditional on
the shared orbital parameters. Supply physical stellar and planet masses
consistently, and pass `transit_orbit_idx` when necessary. For RV-only inference,
use `NbodyRV.get_rv` and omit the timing likelihood.

The likelihood blocks here and below can also be placed in helper functions
and called from inside the model if you want to reuse them; that is optional.

## Photometric likelihoods

`NbodyTransit.get_flux` returns a flux perturbation with an out-of-transit
baseline of **zero**; see [Photodynamics](models.md#photodynamics) for the
forward calculation and additional orbital, radius, and limb-darkening inputs.
Here `flux_obs` is normalized observed flux **minus one**, and `flux_error` is
its uncertainty in the same relative-flux units. Prepare these data arrays
outside the model, matching the light-curve times configured in `nt`.

For independent errors, add the following **inside the model**, after
`flux_zero, tc = nt.get_flux(par)`:

```python
# Inside the NumPyro model, after computing flux_zero.
meanflux = numpyro.sample("meanflux", dist.Uniform(-1e-4, 1e-4))
jitter_flux = numpyro.sample("jitter_flux", dist.HalfNormal(1e-4))
sigma_flux = jnp.sqrt(jnp.asarray(flux_error)**2 + jitter_flux**2)
numpyro.sample(
    "flux_obs",
    dist.Normal(flux_zero + meanflux, sigma_flux),
    obs=jnp.asarray(flux_obs),
)
```

Here `meanflux` is a fitted baseline offset, not a transit-depth parameter.
Choose its prior and the jitter prior for your normalization and instrument.
The predicted `tc` need not have a separate observed likelihood. In particular,
timings extracted from the **same light curve** are not independent additional
data; multiplying their likelihood by the light-curve likelihood would
generally count information twice.

### Correlated photometric noise with a Gaussian process

The [Kepler-51 photodynamics notebook](https://github.com/kemasuda/jnkepler/blob/main/examples/kep51_photodynamics_gp.ipynb)
uses a Matérn-3/2 Gaussian process (GP) for correlated residuals. Its
**transit parameter optimization** and **running HMC** sections show how the
noise model is combined first with fixed, then with free orbital parameters.
Import `tinygp` during session setup:

```python
import tinygp
from tinygp.kernels import quasisep as qk
```

Inside the model, **replace the entire independent-noise block above**, including
its baseline and jitter priors, with this GP block:

```python
# Inside the NumPyro model, after computing flux_zero.
meanflux = numpyro.sample("meanflux", dist.Uniform(-1e-4, 1e-4))
lna = numpyro.sample("lna", dist.Uniform(-14.0, -4.0))
lnc = numpyro.sample("lnc", dist.Uniform(-5.0, 1.0))
lnjitter = numpyro.sample("lnjitter", dist.Uniform(-14.0, -4.0))
kernel_gp = qk.Matern32(sigma=jnp.exp(lna), scale=jnp.exp(lnc))
gp = tinygp.GaussianProcess(
    kernel_gp, jnp.asarray(times_lc),
    diag=jnp.asarray(flux_error)**2 + jnp.exp(2.0 * lnjitter),
    mean=0.0,
)
residual = jnp.asarray(flux_obs) - (flux_zero + meanflux)
numpyro.sample("flux_obs", gp.numpyro_dist(), obs=residual)
```

Use the same sorted `times_lc` as in `nt.set_lcobs`, with matching fluxes and
errors. `exp(lna)` is a
relative-flux amplitude; `exp(lnc)` is a correlation timescale in **days** when
the inputs are days, despite the compact historical name `lnc`. The `diag`
argument contains **variances**, not standard deviations. These prior ranges
follow the example and should be reconsidered for other data. The GP likelihood
includes both the correlated residual term and the covariance normalization;
it is not just a smoothed residual correction. GP parameters can trade off with
transit parameters, so inspect their posterior as well as the transit fit.

These blocks add the measurement models. An extended model must also
declare priors or fixed values for all its forward-model inputs and initialize
its additional sample sites. The photodynamics notebook gives a complete
example with staged optimization before NUTS.

Next: [initialize, tune, and diagnose NUTS](nuts.md) for the original Gaussian
TTV model, or explore the [additional forward models](models.md).
