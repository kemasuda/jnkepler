import numpy as np
import numpy.testing as npt

import jax.numpy as jnp
from jax import grad, jacfwd, jacrev

from jnkepler.jaxttv.symplectic import solve_dE, kepler_step, integrate_xv
from jnkepler.jaxttv.conversion import G


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


def _circular_system(nplanet, nstep=200, dt=1.0):
    """Jacobi coordinates for a star plus nplanet circular coplanar planets."""
    periods = [20.0, 35.0, 55.0, 80.0, 120.0][:nplanet]
    masses = jnp.array([1.0] + [3e-6] * nplanet)

    x = np.zeros((nplanet, 3))
    v = np.zeros((nplanet, 3))
    for i, period in enumerate(periods):
        gm = float(G * jnp.sum(masses[:i + 2]))
        a = (gm * (period / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
        x[i, 0] = a
        v[i, 1] = np.sqrt(gm / a)

    times = jnp.arange(0.0, nstep * dt, dt)
    return jnp.array(x), jnp.array(v), masses, times


def test_integrate_xv_reverse_matches_forward_above_policy_threshold():
    """Reverse-mode gradients survive the checkpoint policy integrate_xv picks.

    With more than four masses, integrate_xv checkpoints its scan step under
    dots_saveable, which changes what the backward pass recomputes rather
    than what it computes. Forward mode is unaffected by that choice, so it
    serves as an independent reference.

    The tolerance is loose because forward and reverse accumulate roundoff in
    different orders over the integration, so they agree to roughly ten
    digits rather than exactly. A policy that corrupted rematerialization
    would fail this by orders of magnitude, not by ulps.
    """
    x0, v0, masses, times = _circular_system(4)
    assert len(masses) > 4, "system must be large enough to select the policy"

    def loss(m):
        _, xv = integrate_xv(x0, v0, m, times, nitr=10)
        return jnp.sum(xv ** 2)

    g_rev = grad(loss)(masses)
    g_fwd = jacfwd(loss)(masses)

    assert np.all(np.isfinite(np.asarray(g_rev)))
    npt.assert_allclose(np.asarray(g_rev), np.asarray(g_fwd),
                        rtol=1e-8, atol=1e-8)


def test_integrate_xv_reverse_matches_forward_below_policy_threshold():
    """The same identity holds below the threshold, as a control."""
    x0, v0, masses, times = _circular_system(3)
    assert len(masses) <= 4

    def loss(m):
        _, xv = integrate_xv(x0, v0, m, times, nitr=10)
        return jnp.sum(xv ** 2)

    npt.assert_allclose(np.asarray(grad(loss)(masses)),
                        np.asarray(jacfwd(loss)(masses)),
                        rtol=1e-8, atol=1e-8)
