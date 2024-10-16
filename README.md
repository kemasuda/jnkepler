# jnkepler
[JAX](https://jax.readthedocs.io/en/latest/index.html) code for modeling nearly Keplerian orbits. Compatible with the No-U-Turn sampler (NUTS) in [NumPyro](https://num.pyro.ai).

- jnkepler.jaxttv: a differentialble model for transit timing variations (TTVs)
- jnkepler.nbodytransit: a differentialble photodynamical model 
- jnkepler.nbodyrv: a differentiable RV model taking into account mutual interactions between planets



## Installation

```python setup.py install```

* requirements: jax, numpyro (for NUTS), [exoplanet-core[jax]](https://github.com/exoplanet-dev/exoplanet-core) (for photodynamical modeling with nbodytransit), [jaxopt](https://jaxopt.github.io/stable/) (for optimization)

  


## Examples

see notebooks in examples


## Applications

- TOI-1136: TTV modeling of 6-planets in a resonance chain [[paper]](https://ui.adsabs.harvard.edu/abs/2022arXiv221009283D/abstract)
- TOI-2015: joint TTV & RV modeling of a two-planet system [[paper]](https://arxiv.org/abs/2310.11775)
- Kepler-51: four-planet modeling including JWST data [[paper]](https://arxiv.org/abs/2410.01625) [[repository]](https://github.com/kemasuda/kep51_jwst)

