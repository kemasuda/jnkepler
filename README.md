# jnkepler
[JAX](https://jax.readthedocs.io/en/latest/index.html) code for modeling nearly Keplerian orbits. Compatible with the No-U-Turn sampler (NUTS) in [numpyro](https://num.pyro.ai).

- jnkepler.jaxttv: a differentialble model for transit timing variations (TTVs)
- jnkepler.nbodytransit: a differentialble photodynamical model 
- jnkepler.nbodyrv: a differentiable RV model taking into account mutual interactions between planets



## Installation

```python setup.py install```

* requirements: jax, numpyro, [exoplanet-core[jax]](https://github.com/exoplanet-dev/exoplanet-core) (for photodynamical modeling with nbodytransit)

  


## Examples

see notebooks in examples
