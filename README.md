# jnkepler

*A differentiable N-body model for multi-planet systems.*

`jnkepler` is a Python package for modeling photometric and radial velocity data of multi-planet systems via N-body integration. Built with [JAX](https://jax.readthedocs.io/en/latest/index.html), it leverages automatic differentiation for efficient computation of model gradients. This enables seamless integration with gradient-based optimizers and Hamiltonian Monte Carlo methods, including the No-U-Turn Sampler (NUTS) in [NumPyro](https://num.pyro.ai). The package is particularly suited for efficiently sampling from multi-planet posteriors involving a large number of parameters and strong degeneracy.

## Subpackages

- **jnkepler.jaxttv**: A differentialble N-body model for analyzing transit timing variations (TTVs) and radial velocities (RVs) of multi-planet systems.
- **jnkepler.nbodytransit**: A differentialble photodynamical model. [`jaxoplanet`](https://jax.exoplanet.codes/en/latest/) needs to be installed for using this package.
- **jnkepler.nbodyrv**: A differentiable RV model taking into account mutual interactions between planets.

See [readthedocs](https://jnkepler.readthedocs.io/en/stable/) for more details.

## Installation

```pip install jnkepler```

### *CPU performance note*

If you use jnkepler on CPU with JAX ≥0.4.32, the default *thunk runtime* in the CPU backend can make computations much slower, especially when computing gradients. 

To avoid this, disable the thunk runtime by setting the following environment variable **before importing jax**:

```bash
export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false"
```

Or inside Python:

```python
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
import jax
```

If this is not done, `jnkepler` will issue a warning on import.


## Examples

Explore example notebooks in the `examples/` directory to see `jnkepler` in action:

- **minimal example**: `examples/minimal_example.ipynb`
  - computing transit times and RVs
  - plotting TTVs
  - adding a non-transiting planet

- **TTV modeling**: `examples/kep51_ttv_normal.ipynb` 
  - posterior sampling with NUTS
  - reproducing the result in [Libby-Roberts et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020AJ....159...57L/abstract)
- **Photodynamical modeling**: `examples/kep51_photodynamics_gp.ipynb`
  - SVI optimization & posterior sampling with NUTS
  - noise modeling using Gaussian Process with [`tinygp`](https://tinygp.readthedocs.io/en/stable/)

## Applications

- TOI-1136: TTV modeling of 6-planets in a resonance chain [[paper]](https://ui.adsabs.harvard.edu/abs/2022arXiv221009283D/abstract)
- TOI-2015: joint TTV & RV modeling of a two-planet system [[paper]](https://arxiv.org/abs/2310.11775)
- Kepler-51: four-planet modeling including JWST data [[paper]](https://arxiv.org/abs/2410.01625) [[repository]](https://github.com/kemasuda/kep51_jwst)
- K2-19: TTVs confirm 3:2 resonance [[paper]](https://arxiv.org/abs/2509.18031)

## References

- Masuda et al. (2024), [A Fourth Planet in the Kepler-51 System Revealed by Transit Timing Variations](https://ui.adsabs.harvard.edu/abs/2024AJ....168..294M/abstract), AJ 168, 294
- Masuda (2025), [jnkepler: Differentiable N-body model for multi-planet systems](https://ui.adsabs.harvard.edu/abs/2025ascl.soft05006M/abstract),  Astrophysics Source Code Library, ascl:2505.006.

