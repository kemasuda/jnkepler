# 4. Models and extensions

The [Quickstart](quickstart.md) fits transit times for three observed planets.
jnkepler also predicts the effects of unseen companions, stellar radial
velocities, and transit light curves. This page introduces those forward
calculations; [the NumPyro guide](numpyro_model.md) covers their likelihoods.

| Goal | Interface | Example to read |
| --- | --- | --- |
| Include a planet without observed transits | `JaxTTV` with `transit_orbit_idx` | Minimal example, **adding non-transiting planets** |
| Predict TTVs and RVs together | `JaxTTV.get_transit_times_and_rvs_obs` | Minimal example, **computing stellar RVs as well** |
| Predict RVs without timing data | `NbodyRV.get_rv` | Same orbital conventions as the minimal example |
| Model transit light curves with interacting orbits | `NbodyTransit` | Kepler-51 photodynamics notebook |
| Calculate independent two-body orbits | `jnkepler.keplerian` functions | Code below and the Keplerian API reference |

For the N-body snippets, continue after the Quickstart. Make a separate
dictionary so that trying an extension does not change the fitted `popt` used
in parts 2–3:

```python
orbit_keys = ["period", "ecosw", "esinw", "tic", "pmass"]
base_par = {key: jnp.asarray(popt[key]) for key in orbit_keys}
base_par["smass"] = 1.0
```

Each section starts from this three-planet model independently. The numerical
values added below illustrate the interfaces; they are not new fits to
Kepler-51. Keep the [orbital conventions and reference epoch](quickstart.md#specify-the-orbital-parameters)
consistent when changing observables.

## Non-transiting planets

An N-body model can contain more planets than the list of observed transit
times. Supply orbital arrays for **all** planets, ordered from inner to outer,
and use `transit_orbit_idx` to map each entry of `tcobs` to its zero-based orbital
index. Here a fourth, outer planet perturbs the three observed planets:

```python
outer_planet = {
    "period": 260.0,
    "ecosw": 1e-3,  # Keep eccentricity nonzero for gradient-based initialization.
    "esinw": 0.0,
    "tic": 200.0,
    "pmass": 3e-6,
}
par_four = {
    key: jnp.concatenate([base_par[key], jnp.array([outer_planet[key]])])
    for key in orbit_keys
}
par_four["smass"] = base_par["smass"]
par_four["cosi"] = jnp.array([0.0, 0.0, 0.0, 0.1])
par_four["lnode"] = jnp.zeros(4)
transit_orbit_idx = jnp.array([0, 1, 2])

tc_four, energy_four = jttv.get_transit_times_obs(
    par_four, transit_orbit_idx=transit_orbit_idx,
)
tc_four_all = jttv.get_transit_times_all_list(
    par_four, transit_orbit_idx=transit_orbit_idx, truncate=False,
)
jttv.plot_model(tc_four_all)
plt.show()
```

The timing output still has 53 entries corresponding to the three observed
planets. All four planets contribute to the dynamics. If the unobserved planet
were between the first and second observed planets, the mapping would instead
be `[0, 2, 3]`, with every orbital array reordered accordingly.

The index selects which planets have timing predictions matched to data; it
does not itself constrain transit geometry. Here we separately tilted the
outer orbit through `cosi`. Its `tic` is an orbital-phase parameter even without
an observed transit. Include all planets in parameter priors and initial values,
and pass the mapping inside a NumPyro model too. The three-planet optimization
bounds from the Quickstart must be extended or replaced before fitting this
four-planet model.

The [minimal example](https://github.com/kemasuda/jnkepler/blob/main/examples/minimal_example.ipynb),
under **adding non-transiting planets**, demonstrates the orbital-index mapping.
As in the Quickstart, the full timing sequence is for plotting; use the
observation-matched output for residuals and likelihoods.

## Radial velocities

### TTVs and RVs from the same integration

RV amplitudes require a physical mass scale. The Quickstart's `pmass` values
are mass ratios because its stellar mass was fixed to one. Adopt a stellar
mass and convert the planet masses consistently:

```python
stellar_mass_solar = 1.0  # Adopted physical stellar mass for this illustration.
par_rv = dict(base_par)
par_rv["smass"] = stellar_mass_solar
par_rv["pmass"] = base_par["pmass"] * stellar_mass_solar
times_rv = np.linspace(500.0, 2500.0, 500)

tc_with_rv, rv_model, energy_with_rv = jttv.get_transit_times_and_rvs_obs(
    par_rv, times_rv,
)
plt.figure()
plt.plot(times_rv, rv_model)
plt.xlabel("Time (days, data reference)")
plt.ylabel("Stellar RV (m/s)")
plt.show()
```

The returned RV is in **m/s**, positive when the star moves away. The timing
array retains the observation order from part 1. `times_rv` need not coincide
with transits but must be inside the integration's supported interpolation
range; leave a margin at its edges. Use the same time reference for both
datasets. The method also accepts `transit_orbit_idx` for models with unobserved
planets.

These are dynamical stellar velocities. Instrument offsets, additional scatter,
and the measurement likelihood belong in the [NumPyro model](numpyro_model.md#rv-likelihoods-and-joint-inference).
See **computing stellar RVs as well** in the
[minimal example](https://github.com/kemasuda/jnkepler/blob/main/examples/minimal_example.ipynb)
for the joint forward calculation.

### RV-only N-body calculations

When no transit times are available, use `NbodyRV`. It needs the integration
interval and orbital parameters, without a `tcobs` list or transit mapping:

```python
from jnkepler.nbodyrv import NbodyRV

nrv = NbodyRV(t_start, t_end, dt, nitr_kepler=jttv.nitr_kepler)
rv_only = nrv.get_rv(times_rv, par_rv)
```

Here `par_rv` and `times_rv` are from the preceding RV snippet. The orbital
elements remain defined at `t_start`, masses are in solar units, and the output
is in m/s. Set `cosi` and `lnode` explicitly for a different geometry. Check
RV precision by comparing finer integration settings relative to the RV
uncertainties; the TTV precision helper does not test RV accuracy.

## Photodynamics

`NbodyTransit` combines interacting orbital dynamics with transit light curves.
Install the additional `jaxoplanet` dependency before using this section.
The [Kepler-51 photodynamics notebook](https://github.com/kemasuda/jnkepler/blob/main/examples/kep51_photodynamics_gp.ipynb)
is the complete workflow: **load test data**, **initialize**, **TTV-only
optimization**, **transit parameter optimization**, **optimizing full
photodynamics model**, and **running HMC**. That staged approach gives the
joint fit a useful starting point.

To illustrate the prediction interface, sample short windows around each
planet's first observed transit and supply transit-shape parameters:

```python
from jnkepler.nbodytransit import NbodyTransit

times_lc = np.sort(np.concatenate([
    times[0] + np.linspace(-0.4, 0.4, 81) for times in tcobs
]))
nt = NbodyTransit(
    t_start, t_end, dt, tcobs, p_init,
    errorobs=errorobs, transit_time_method="fast",
)
nt.set_lcobs(
    times_lc,
    exposure_time=29.4 / 1440.0,
    supersample_factor=10,
    overlapping_transit=False,
)
par_photo = dict(base_par)
par_photo.update({
    "srad": 1.0,
    "radius_ratio": jnp.array([0.07, 0.09, 0.10]),
    "b": jnp.array([0.2, 0.3, 0.1]),
    "q1": 0.4,
    "q2": 0.3,
})
flux_zero, tc_photo = nt.get_flux(par_photo)

fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for planet, ax in enumerate(axes):
    window = np.abs(times_lc - tcobs[planet][0]) < 0.5
    ax.plot(times_lc[window] - tcobs[planet][0], 1.0 + flux_zero[window])
    ax.set_xlabel("Time from observed transit (days)")
axes[0].set_ylabel("Normalized flux")
fig.tight_layout()
plt.show()
```

`set_lcobs` configures evaluation times and exposure averaging; it does not take
the measured fluxes or their uncertainties. Exposure time is in days.
`supersample_factor` controls the grid within each exposure; compare finer
grids when checking light-curve precision. The code uses positions and velocities
near the supplied transit centers, so prepare transit windows and a timing
list covering their events, as in the notebook.

| Additional input | Convention |
| --- | --- |
| `srad` | Stellar radius in solar radii when using a physical stellar mass |
| `radius_ratio` | One planet-to-star radius ratio per planet |
| `b` or `cosi` | Impact parameters converted to inclinations, or inclinations supplied directly through their cosines |
| `q1`, `q2` or `u1`, `u2` | Two-parameter form of quadratic limb darkening, with `q1`, `q2` in [0, 1], or the coefficients directly |
| `lnode` | Ascending-node longitudes in radians; zero by default |

When fitting `lnode`, fix one planet's ascending-node longitude (for example,
to zero): rotating the entire system about the line of sight leaves the
photometric model unchanged, so only relative node longitudes can be
constrained. Leaving every `lnode` free introduces an unconstrained common
rotation. The Kepler-51 notebook fixes the outermost planet's `lnode` to zero
and fits the inner planets' node longitudes relative to it.

The illustrative `b` values change the geometry from the edge-on TTV fit;
these are not optimized light curves. With the mass normalized to `smass=1`,
the inferred `srad` is a scaled radius: physically
$(R_\star/R_\odot)/(M_\star/M_\odot)^{1/3}$. Alternatively, the interface accepts
`sdens` in solar-density units; with no supplied mass it sets `smass=1` and
derives `srad`. Do not mix this normalization with physical RV amplitudes.

**`get_flux` returns a zero-baseline flux perturbation**, so the plot adds one.
Its second output is the flattened transit-time prediction. To predict RVs as
well, `nt.get_flux_and_rv(par_photo, times_rv)` returns `(flux_zero, tc, rv)`;
use physical stellar and planet masses consistently for that calculation.
The [photometric likelihood section](numpyro_model.md#photometric-likelihoods)
explains baseline offsets, independent noise, and GP residuals. The notebook's
**plot light-curve models** section shows posterior light curves and GP predictions.

The current `NbodyTransit` interface does **not support non-transiting
companions**. `overlapping_transit=True` allows simultaneous transits by adding
their flux losses, but does not model planets occulting one another.

Also use `overlapping_transit=True` for closely spaced transits, even when their
actual transit intervals do not overlap. With `False`, each exposure subsample
is assigned only to the nearest **observed transit center**. A long transit can
extend past the midpoint to a neighboring shorter transit and lose part of its
signal under that assignment. Use `False` only when every possible transit
signal stays within its own center's nearest-neighbor interval, allowing for
timing shifts, duration changes, and exposure width over the parameter range
being explored. If this separation is uncertain, use `True`.

## Pure Keplerian orbits

For independent two-body motion without planet–planet perturbations, use
`jnkepler.keplerian`. These are functions rather than a `JaxTTV`-style class.
They do not produce interaction-driven TTVs. The phase parameter here is
`tau`, the time of periastron passage; `tic_to_tau` converts a conjunction epoch
using the convention $\omega+f=\pi/2$.

### RV shapes and amplitudes

```python
from jnkepler.keplerian import radial_velocity_shape_multi, tic_to_tau

times_kep = jnp.linspace(500.0, 2500.0, 500)
kepler_par = {
    "period": jnp.array([45.0, 85.0]),
    "ecc": jnp.array([0.05, 0.03]),
    "omega": jnp.array([0.4, 1.0]),
}
kepler_par["tau"] = tic_to_tau(
    jnp.array([159.0, 210.0]),
    kepler_par["period"], kepler_par["ecc"], kepler_par["omega"],
)
rv_shapes = radial_velocity_shape_multi(times_kep, kepler_par)
semi_amplitude = jnp.array([3.0, 5.0])  # m/s, supplied separately.
gamma = 0.0  # m/s.
rv_kepler = gamma + jnp.sum(semi_amplitude[:, None] * rv_shapes, axis=0)
```

`radial_velocity_shape_multi` returns a **dimensionless** array with shape
`(number_of_planets, number_of_times)`, containing
$\cos(\omega+f)+e\cos\omega$. Supply semi-amplitudes and sum the contributions
to obtain m/s; the function does not convert masses into amplitudes.
`radial_velocity_shape` is the single-orbit version. Angles are radians, and
periods, `tau`, and evaluation times use days with a consistent reference.

### Positions and velocities

```python
from jnkepler.keplerian import elements_to_xv

state_par = {key: value[0] for key, value in kepler_par.items()}
state_par.update({"inc": jnp.pi / 2, "lnode": 0.0, "mass": 1.0})
state = elements_to_xv(times_kep, state_par)
print(state["x"].shape, state["v"].shape)
```

For this single orbit the shapes are `(number_of_times, 3)`. `mass` is the
**total two-body mass** in solar masses; positions are in AU and velocities
in AU/day. These are relative orbital state vectors, not the star's reflex
position or velocity. The multiple-orbit output has shape
`(number_of_times, number_of_orbits, 3)`.

`elements_to_xv_scaled` instead returns position divided by semimajor axis
(dimensionless) and velocity divided by semimajor axis (1/day).
`xv_to_elements` provides the inverse conversion; supply `t_ref` when you need
the periastron epoch. See the [Keplerian API reference](api/jnkepler.keplerian.rst)
for the full signatures. There is no dedicated Keplerian notebook in the
current [example collection](examples.md); the snippets above provide the
starting point.
