# jnkepler

*A differentiable N-body model for multi-planet systems.*

`jnkepler` is a Python package for modeling photometric and radial velocity data of multi-planet systems via N-body integration. Built with [JAX](https://jax.readthedocs.io/en/latest/index.html), it leverages automatic differentiation for efficient computation of model gradients. This enables seamless integration with gradient-based optimizers and Hamiltonian Monte Carlo methods, including the No-U-Turn Sampler (NUTS) in [NumPyro](https://num.pyro.ai). The package is particularly suited for efficiently sampling from multi-planet posteriors involving a larger number of parameters and strong degeneracy.

## Subpackages

- **jnkepler.jaxttv**: A differentialble N-body model for analyzing transit timing variations (TTVs) and radial velocities (RVs) of multi-planet systems.
- **jnkepler.nbodytransit**: A differentialble photodynamical model. [`jaxoplanet`](https://jax.exoplanet.codes/en/latest/) needs to be installed for using this package.
- **jnkepler.nbodyrv**: A differentiable RV model taking into account mutual interactions between planets.

See [readthedocs](https://jnkepler.readthedocs.io/en/latest/index.html) for more details.

## Installation

```python setup.py install```


## Examples

Explore example notebooks in the `examples/` directory to see `jnkepler` in action:

- **minimal example for computing transit times**: `examples/minimal_example.ipynb`

- **TTV modeling**: `examples/kep51_ttv_iidnormal.ipynb` 
  - posterior sampling with NUTS
  - reproducing the result in [Libby-Roberts et al. 2020](https://ui.adsabs.harvard.edu/abs/2020AJ....159...57L/abstract)
- **Photodynamical modeling**: `examples/kep51_photodynamics_gp.ipynb`
  - SVI optimization & posterior sampling with NUTS
  - noise modeling using Gaussian Process with [`celerite2.jax`](https://celerite2.readthedocs.io/en/latest/api/jax/)



## Applications

- TOI-1136: TTV modeling of 6-planets in a resonance chain [[paper]](https://ui.adsabs.harvard.edu/abs/2022arXiv221009283D/abstract)
- TOI-2015: joint TTV & RV modeling of a two-planet system [[paper]](https://arxiv.org/abs/2310.11775)
- Kepler-51: four-planet modeling including JWST data [[paper]](https://arxiv.org/abs/2410.01625) [[repository]](https://github.com/kemasuda/kep51_jwst)

## References

- Masuda et al. (2024), [A Fourth Planet in the Kepler-51 System Revealed by Transit Timing Variations](https://arxiv.org/abs/2410.01625), AJ in press.

