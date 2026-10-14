import numpy as np
import numpy.testing as npt
import pytest

import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, jit

from jnkepler.jaxttv.symplectic import solve_dE, kepler_step
from jnkepler.jaxttv import symplectic as symp
from jnkepler.jaxttv.utils import initialize_jacobi_xv


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


@pytest.mark.parametrize("h", [0., 0.06, -0.3])
def test_forward_corrector_ttvfast_operation_order(monkeypatch, h):
    """Record primitive calls, so ordering/sign errors cannot hide numerically."""
    calls = []

    def drift(x, v, ki, dt):
        calls.append(("D", float(dt)))
        return x, v

    def kick(x, v, ki, masses, dt):
        calls.append(("K", float(dt)))
        return x, v

    monkeypatch.setattr(symp, "kepler_step", drift)
    monkeypatch.setattr(symp, "nbody_kicks", kick)
    symp.real_to_mapTO(None, None, None, None, h)
    a = np.sqrt(7./40.)*h
    b = h/(96.*np.sqrt(7./40.))
    # Original TTVFast: Z(a2,b2) then Z(a1,b1), each C(-a,-b), C(a,b).
    expected = [("D", a), ("K", -b), ("D", -a),
                ("D", -a), ("K", b), ("D", a),
                ("D", -a), ("K", b), ("D", a),
                ("D", a), ("K", -b), ("D", -a)]
    assert [op for op, _ in calls] == [op for op, _ in expected]
    npt.assert_allclose([dt for _, dt in calls], [dt for _, dt in expected],
                        rtol=0., atol=1e-17)


def test_forward_corrector_ttvfast_coefficients():
    alpha = np.sqrt(7./40.)
    beta = 1./(48.*alpha)
    npt.assert_allclose(symp.compute_corrector_coefficientsTO(),
                        [-alpha, alpha, -beta/2, beta/2], rtol=0., atol=1e-17)


@pytest.fixture(scope="module")
def corrector_state():
    # Interacting, eccentric, inclined orbits, shared with the original experiment.
    params = {key: jnp.asarray(value) for key, value in {
        "period": [2.1, 3.4, 5.7], "ecosw": [0.03, -0.04, 0.01],
        "esinw": [0.02, 0.01, -0.03], "cosi": [0.002, -0.001, 0.003],
        "lnode": [0.01, -0.02, 0.04], "tic": [0.6, 0.9, 1.3],
        "pmass": [2e-4, 3e-4, 1e-4],
    }.items()}
    x, v, masses = initialize_jacobi_xv(params, 0.)
    return x, v, masses, symp._compute_ki(masses)


@pytest.mark.parametrize("h", [0., 0.06, 0.3, -0.3])
def test_forward_corrector_matches_direct_ttvfast_sequence(corrector_state, h):
    x, v, masses, ki = corrector_state
    # Independent primitive sequence and coefficients, without C or Z helpers.
    a = np.sqrt(7./40.)*h
    b = h/(96.*np.sqrt(7./40.))
    operations = [("D", a), ("K", -b), ("D", -a),
                  ("D", -a), ("K", b), ("D", a),
                  ("D", -a), ("K", b), ("D", a),
                  ("D", a), ("K", -b), ("D", -a)]
    xd, vd = x, v
    for op, dt in operations:
        xd, vd = (symp.kepler_step(xd, vd, ki, dt) if op == "D" else
                  symp.nbody_kicks(xd, vd, ki, masses, dt))
    for transform in (symp.real_to_mapTO, jit(symp.real_to_mapTO)):
        npt.assert_allclose(transform(x, v, ki, masses, h), (xd, vd),
                            rtol=0., atol=2e-16)


def test_forward_corrector_differentiable(corrector_state):
    x, v, masses, _ = corrector_state
    state = jnp.concatenate([x.ravel(), v.ravel(), masses, jnp.array([0.12])])
    nx = x.size

    def output(theta):
        xp = theta[:nx].reshape(x.shape)
        vp = theta[nx:2*nx].reshape(v.shape)
        mp, h = theta[2*nx:-1], theta[-1]
        corrected = symp.real_to_mapTO(xp, vp, symp._compute_ki(mp), mp, h)
        return jnp.concatenate([a.ravel() for a in corrected])

    jf = jit(jacfwd(output))(state)
    jr = jit(jacrev(output))(state)
    assert np.isfinite(jf).all() and np.isfinite(jr).all()
    npt.assert_allclose(jf, jr, rtol=2e-9, atol=2e-12)
    # Exercise positions, velocities, masses and timestep together.
    direction = jnp.linspace(-1., 1., len(state))*jnp.maximum(jnp.abs(state), 1e-4)
    delta = 1e-5
    finite_diff = (output(state+delta*direction)-output(state-delta*direction))/(2*delta)
    npt.assert_allclose(jf@direction, finite_diff, rtol=1e-6, atol=2e-10)
