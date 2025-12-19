Quickstart
==========

This page provides a minimal overview of how to start using **jnkepler**.

Examples
-------

Explore example notebooks in the [examples](https://github.com/kemasuda/jnkepler/tree/main/examples) to see `jnkepler` in action:

- **minimal example**: `examples/minimal_example.ipynb`
  - computing transit times and RVs
  - plotting TTVs
  - adding a non-transiting planet

- **TTV modeling**: `examples/kep51_ttv_normal.ipynb` 
  - posterior sampling with NUTS
  - reproducing the result in [Libby-Roberts et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020AJ....159...57L/abstract)

- **Photodynamical modeling**: `examples/kep51_photodynamics_gp.ipynb`
  - SVI optimization & posterior sampling with NUTS
  - noise modeling using Gaussian Process with [tinygp](https://tinygp.readthedocs.io/en/stable/)