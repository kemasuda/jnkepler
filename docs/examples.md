# 5. Choose an example notebook

After the [three-part tutorial](quickstart.md), choose a notebook close to your
analysis. The notebooks contain complete workflows and system-specific choices;
the tutorials explain the shared steps. [Models and extensions](models.md)
introduces non-transiting planets, RVs, photodynamics, and pure Keplerian
calculations, with links to relevant notebook sections. Read each notebook's
opening notes before adopting its priors, initialization, or runtime settings.

Download or clone the repository and launch notebooks from `examples/`, keeping
the accompanying data files there. Install jnkepler as described in
[Installation](install.md). The inference notebooks also import `seaborn` for
plot styling; the photodynamical example needs `jaxoplanet` and `tinygp`, and
the V1298 Tau notebook imports `dill` in its resampling section.

| Notebook | What to learn | Scope and cautions |
| --- | --- | --- |
| [Minimal example](https://github.com/kemasuda/jnkepler/blob/main/examples/minimal_example.ipynb) | Transit times, TTV plots, stellar RVs, and a non-transiting planet | Forward calculations without posterior sampling |
| [Kepler-51: Gaussian TTV likelihood](https://github.com/kemasuda/jnkepler/blob/main/examples/kep51_ttv_normal.ipynb) | Optimization, an eccentricity-prior factor, Fisher initialization, and NUTS | Three-planet analysis based on [Libby-Roberts et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020AJ....159...57L/abstract) and reproduced in [Masuda et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024AJ....168..294M/abstract); does not include the later fourth planet |
| [V1298 Tau: Student's t likelihood](https://github.com/kemasuda/jnkepler/blob/main/examples/v1298tau_ttv_student.ipynb) | Fitted noise tails and scale, mass priors, and posterior reweighting | Based on [Livingston et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026Natur.649..310L/abstract); uses a modified lower mass-prior bound for sampling efficiency, as explained in the notebook |
| [TRAPPIST-1: Student's t likelihood](https://github.com/kemasuda/jnkepler/blob/main/examples/trappist-1_ttv_student.ipynb) | A seven-planet fit and comparison with [Agol et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021PSJ.....2....1A/abstract) | More demanding than the introductory example; uses the included `T1_agol2021_thin30.nc` reference samples for the final comparison |
| [Kepler-51: photodynamics with a Gaussian process](https://github.com/kemasuda/jnkepler/blob/main/examples/kep51_photodynamics_gp.ipynb) | Light-curve modeling, staged optimization, and GP noise | Based on the photodynamical analysis in the appendix of [Masuda et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024AJ....168..294M/abstract); requires `jaxoplanet` and `tinygp` and also omits the later fourth planet |

The notebook introductions report approximate full-run times on an Apple M4 Max
with jnkepler 0.2.8 and JAX 0.11.2: 6 minutes for Gaussian Kepler-51 TTVs,
9 hours for V1298 Tau, 4.5 hours for TRAPPIST-1, and 1.5 hours for Kepler-51
photodynamics. These are reported example timings, not guarantees for your
hardware or evidence that a shorter run has converged.

For the main tutorial path, read the Kepler-51 Gaussian notebook in this order:
**define tcobs, errorobs, p_init** → **optimization** → **check precision &
residual distribution** → **setup & run HMC** → posterior models and diagnostic
plots. The Quickstart covers the first three steps, the NumPyro guide explains
the statistical model, and the NUTS guide covers sampling and diagnostics.
The tutorial includes the notebook's eccentricity-prior correction, explained
in [the prior discussion](numpyro_model.md#understand-the-prior-before-changing-coordinates).
The notebook also uses Fisher-based initialization; the
[mass-matrix appendix](mass_matrix.md) explains that additional step.

When adapting an example, revisit the data's time reference and planet order,
the parameterization and priors, numerical timing accuracy, and sampler
diagnostics. The [NumPyro guide](numpyro_model.md) explains changes to the
probability model; the [NUTS guide](nuts.md) explains sampling checks.
