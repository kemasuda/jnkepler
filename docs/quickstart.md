# 1. Quickstart: fit transit times

**Try the Kepler-51 example.** The accompanying
[Kepler-51 notebook](https://github.com/kemasuda/jnkepler/blob/main/examples/kep51_ttv_normal.ipynb)
reports a full runtime of about **6 minutes**, including optimization and NUTS
posterior sampling, on an Apple M4 Max with jnkepler 0.2.10 and JAX 0.12.0.1.

This three-part tutorial walks through that workflow: load the timings, optimize
an orbital model, [write a NumPyro model](numpyro_model.md), and [run
NUTS](nuts.md). Here we reach a fitted model and check its residuals and numerical
precision. Run the Python blocks in order in one notebook or Python session,
after following [Installation](install.md). The tutorial uses simpler sampler
initialization; the reported runtime belongs to the linked notebook.

We use the three-planet Kepler-51 example shipped with jnkepler. It is a teaching
example based on the older timing dataset, not a fit of the four-planet system
with the later observations. See [the example notebooks](examples.md) for the
scientific context.

## Set up the session

With both JAX and jaxlib >=0.12, no CPU performance workaround is needed.
For older versions, set `XLA_FLAGS` **before importing any library that imports JAX**.
Configure CPU devices before the first JAX computation. In an already-used notebook,
restart the kernel first. The version-specific performance settings below
follow the [README](https://github.com/kemasuda/jnkepler#installation), preserving
any existing performance options in `XLA_FLAGS`.

```python
import os
from importlib.metadata import version
from packaging.version import Version

jax_release = Version(version("jax")).release
jaxlib_release = Version(version("jaxlib")).release
if "XLA_FLAGS" not in os.environ and min(jax_release, jaxlib_release) < (0, 12):
    if jax_release >= (0, 7, 0):
        os.environ["XLA_FLAGS"] = (
            "--xla_backend_extra_options="
            "xla_cpu_small_while_loop_byte_threshold=65536"
        )
    elif jax_release >= (0, 4, 32):
        os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpyro
from importlib.resources import files

jax.config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")
num_chains = 4
numpyro.set_host_device_count(num_chains)
print("JAX device count:", jax.local_device_count())

from jnkepler.jaxttv import JaxTTV
```

Use 64-bit calculations throughout the tutorial. The first model evaluation
includes JAX compilation; later calls with the same shapes can be much faster.

As in the Kepler-51 notebook, we configure four CPU devices for the four
parallel NUTS chains in [part 3](nuts.md#run-with-automatic-adaptation).
The printed device count should be four. If it is smaller, restart the kernel
and rerun this setup before evaluating the model.

## Load observations

The data file is included in the installed package, so this code works without
cloning the repository or changing directory.

```python
data_file = files("jnkepler").joinpath("data/kep51_ttv.txt")
with data_file.open("r") as stream:
    data = pd.read_csv(
        stream, sep=r"\s+", header=None,
        names=["epoch", "time", "error", "dataset", "planet"],
    )

tcobs = []
errorobs = []
for planet in (1, 2, 3):
    rows = data.loc[data["planet"] == planet].sort_values("time")
    tcobs.append(rows["time"].to_numpy())
    errorobs.append(rows["error"].to_numpy())

p_init = [45.155305, 85.31646, 130.17809]
print("Number of timings per planet:", [len(t) for t in tcobs])
```

`tcobs` is a list with one one-dimensional array per planet; different planets
can have different numbers of observations. `errorobs` has the same structure
and contains positive timing uncertainties in **days**. Keep the same planet
order everywhere: here it is inner to outer. This dataset contains 31, 11, and
11 timings, for a total of 53.

For your own data, use a consistent time standard and reference offset for all
observations, integration times, and conjunction epochs. jnkepler does not
convert between time standards. `p_init` provides approximate mean periods for
assigning transit epochs across gaps. Check those assignments if the periods
are poorly known; they are distinct from the orbital `period` parameters below.

## Initialize the integrator

```python
t_start, t_end, dt = 155.0, 2950.0, 1.0
jttv = JaxTTV(
    t_start, t_end, dt, tcobs, p_init,
    errorobs=errorobs, transit_time_method="fast",
)
```

All three integration settings are in days. The interval must enclose the
observations with room around its edges. `dt` controls the N-body integration
step; it is not the spacing between observations. The value above is a starting
choice for this example, to be checked below.

`fast` finds transits near the observed times efficiently. It is intended for
models reasonably close to the data. Use `transit_time_method="newton"` when
constructing a model that needs more robust transit finding away from a good
solution. Build a new `JaxTTV` object when changing the integration setup or
solver after JIT compilation; this avoids relying on mutations of a cached
object. Transit matching still needs physically sensible periods and epochs.

## Specify the orbital parameters

```python
par_dict = {
    "smass": 1.0,
    "pmass": jnp.array([9.9e-6, 1.2e-5, 1.6e-5]),
    "period": jnp.array([45.154107, 85.314483, 130.1841370]),
    "ecosw": jnp.array([-0.0172, 0.0189, 0.0117]),
    "esinw": jnp.array([-0.0407, -0.0348, -0.0281]),
    "tic": jnp.array([159.1103, 295.3210, 212.0234]),
}
```

The arrays contain one entry per planet, in the same order as `tcobs`.
These are **osculating Jacobi orbital elements at `t_start`**, using the total
interior mass in the period-to-orbit conversion.

| Key | Meaning | Units / convention |
| --- | --- | --- |
| `smass` | Stellar mass, a scalar | Solar masses; defaults to 1 |
| `pmass` | Planet masses | Solar masses when supplying a physical `smass` |
| `period` | Osculating orbital periods | Days |
| `ecosw`, `esinw` | $e\cos\omega$, $e\sin\omega$ | Dimensionless; $e<1$ |
| `tic` | Times of inferior conjunction specifying orbital phase | Days, with the same time reference as the observations |
| `cosi` | Cosine of inclination | Defaults to 0 (edge-on) |
| `lnode` | Longitude of ascending node | Radians; defaults to 0 |

`tic` parameterizes phase on each osculating orbit. It is not a list of all
predicted transits, and interactions mean it need not exactly equal an observed
transit time. Changing `t_start` while retaining the same elements generally
changes the physical initial condition.

For this TTV-only example we fix `smass=1`. In this normalization, the numerical
`pmass` values represent planet-to-star mass ratios. Converting them to physical
planet masses requires an adopted stellar mass. If you instead supply a
physical `smass`, supply `pmass` in solar masses too; do not mix the conventions.
The TTV-only normalization should not be carried over to absolute RV amplitudes
without accounting for the stellar mass.

## Calculate timings for fitting

Calculate predictions corresponding to the supplied observations:

```python
tcmodel, fractional_energy_error = jttv.get_transit_times_obs(par_dict)
print("Predicted timings:", tcmodel.shape)
print("Fractional energy error:", float(fractional_energy_error))
```

`get_transit_times_obs` returns only the transits corresponding to the supplied
observations. The output concatenates planet 1, then planet 2, then planet 3;
it is **not globally sorted by time**. `jttv.tcobs_flatten` and
`jttv.errorobs_flatten` use that same order. Use these matching arrays when
writing a likelihood or checking residuals of the fitted model.

## Optimize the orbital parameters

Follow the notebook's **optimization** step to find a nearby least-squares
solution before sampling:

```python
from jnkepler.jaxttv.infer import (
    ttv_default_parameter_bounds,
    ttv_optim_least_squares,
)

param_bounds = ttv_default_parameter_bounds(jttv)
pinit = dict(par_dict)
pinit["lnpmass"] = jnp.log(pinit["pmass"])
popt = ttv_optim_least_squares(
    jttv, param_bounds, pinit=pinit,
    jac=True, diff_mode="auto", transit_time_method="newton", plot=False,
)
```

The helper supplies bounds on the periods, conjunction epochs, eccentricity
components, and masses. Inspect them for your own system. At this stage they
constrain the search; [part 2](numpyro_model.md#choose-parameters-and-bounds)
explains their statistical meaning when used as priors.

The optimizer varies log masses and returns both `lnpmass` and `pmass` in
`popt`. This numerical choice does not impose a log-mass prior on a later
NumPyro model. The default `loss="linear"` minimizes ordinary weighted squared
timing residuals. It does not evaluate a NumPyro model or include its priors,
jitter, or Student's t likelihood. The notebook's older function name
`ttv_optim_curve_fit` is a compatibility wrapper for this routine.

Inspect the optimizer's termination report. For another system, use physically
plausible starting points. Without `pinit`, `n_start` tries several starts by
randomizing only log masses; this is not a global search over orbital solutions.

## Check precision and residuals

An energy diagnostic alone does not establish the accuracy of the transit
times. At the fitted solution, compare the nominal calculation with a finer
integration using Newton transit finding:

```python
tc_fit, tc_reference_fit = jttv.check_timing_precision(popt)
residual_sigma = (jttv.tcobs_flatten - tc_fit) / jttv.errorobs_flatten
precision_in_sigma = np.abs(tc_fit - tc_reference_fit) / jttv.errorobs_flatten
print("chi-squared:", float(np.sum(residual_sigma**2)))
print("Largest timing difference / observed uncertainty:", precision_in_sigma.max())

plt.figure()
plt.plot(jttv.tcobs_flatten, residual_sigma, ".")
plt.axhline(0, color="gray")
plt.xlabel("Time (days, data reference)")
plt.ylabel("Residual / observed uncertainty")
plt.show()
```

The default reference step is `p_init[0] * 1e-3`, about 0.045 days here;
`dtfrac` is a fraction of the inner period, not of `dt`. Seek numerical
differences well below the observational errors and check further refinement
when needed. If differences are material, reduce `dt` and/or compare transit
solvers, then refit. Repeat at representative posterior draws later.

Optimization above uses Newton transit finding, while the nominal calculation
uses our object's `fast` solver. This check therefore also tests the numerical
model that will be used for sampling at the optimized parameters. Look for
systematic residual patterns as well as a reasonable chi-squared; see
**check precision & residual distribution** in the
[Kepler-51 notebook](https://github.com/kemasuda/jnkepler/blob/main/examples/kep51_ttv_normal.ipynb)
for additional plots.

## Plot the fitted TTV curves

For a TTV curve that includes unobserved transits between the measurements,
calculate the full sequence and pass it to the built-in plotting function:

```python
tc_all_list = jttv.get_transit_times_all_list(popt, truncate=False)
jttv.plot_model(tc_all_list)
plt.show()
```

`tc_all_list` contains one array per planet. `truncate=False` retains the
sequence over the full integration interval; the default `truncate=True`
cuts off each planet's sequence near its last observed transit.
`plot_model` uses the observations and errors stored in `jttv` and creates
a TTV panel and a residual panel for each planet. By default, it subtracts
a linear ephemeris fitted to the observations to display the TTVs in minutes.

**This full sequence is for visualization, not for fitting.** Its entries
include unobserved transits and do not align with `jttv.tcobs_flatten`.
Use `get_transit_times_obs` for the likelihood in the next tutorial.
Also, `get_transit_times_all_list` always uses Newton transit finding,
independently of `jttv.transit_time_method`. Its plotted residuals may therefore
differ slightly from residuals evaluated with the `fast` fitting model.

The **plot** section of the
[minimal example](https://github.com/kemasuda/jnkepler/blob/main/examples/minimal_example.ipynb)
shows this forward-calculation interface without the optimization workflow.

Next: [write a NumPyro model](numpyro_model.md), keeping `jttv`, `param_bounds`,
and `popt` in the same session. For non-transiting planets, RVs, photodynamics,
and independent Keplerian orbits, see [Models and extensions](models.md).
