import numpy as np
import numpy.testing as npt

import jax.numpy as jnp
from jax import grad, jacfwd, jacrev

from jnkepler.jaxttv.symplectic import solve_dE, kepler_step


def _kepler_residual(dE, ecosE0, esinE0, dM):
    return dE + (1.0 - jnp.cos(dE)) * esinE0 - jnp.sin(dE) * ecosE0 - dM


def test_solve_dE_solves_kepler_equation():
    ecosE0 = jnp.array([0.10, 0.20, -0.05])
    esinE0 = jnp.array([0.05, -0.03, 0.02])
    dM = jnp.array([0.01, 0.20, -0.08])

    dE = solve_dE(ecosE0, esinE0, dM, max_iter=20, tol=1e-14)
    res = _kepler_residual(dE, ecosE0, esinE0, dM)

    npt.assert_allclose(np.asarray(res), 0.0, rtol=0.0, atol=1e-12)


def test_solve_dE_jacfwd_matches_jacrev():
    ecosE0 = jnp.array([0.10, 0.20, -0.05])
    esinE0 = jnp.array([0.05, -0.03, 0.02])
    dM = jnp.array([0.01, 0.20, -0.08])

    f = lambda x: solve_dE(x, esinE0, dM, max_iter=20, tol=1e-14)

    jf = jacfwd(f)(ecosE0)
    jr = jacrev(f)(ecosE0)

    npt.assert_allclose(np.asarray(jf), np.asarray(jr), rtol=1e-10, atol=1e-12)


def test_solve_dE_grad_matches_implicit_formula():
    ecosE0 = 0.12
    esinE0 = -0.07
    dM = 0.30

    f = lambda ece: solve_dE(ece, esinE0, dM, max_iter=20, tol=1e-14)
    dE = f(ecosE0)

    s = jnp.sin(dE)
    c = jnp.cos(dE)
    fp = 1.0 + s * esinE0 - c * ecosE0
    expected = s / fp

    got = grad(f)(ecosE0)

    npt.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-10, atol=1e-12)


def test_kepler_step_jacfwd_runs():
    x = jnp.array([[1.0, 0.0, 0.0]])
    v = jnp.array([[0.0, 0.9, 0.0]])
    gm = jnp.array([1.0])
    dt = 0.05

    f = lambda x_: jnp.sum(kepler_step(x_, v, gm, dt, nitr=20)[0])

    j = jacfwd(f)(x)

    assert j.shape == x.shape
    assert np.all(np.isfinite(np.asarray(j)))
