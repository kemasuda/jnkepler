"""Numerical and structural regressions for saving the converged Kepler root."""

from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv import symplectic as symp
from jnkepler.jaxttv.utils import initialize_jacobi_xv
from jnkepler.tests import read_testdata_tc


def _integrate_baseline(x, v, masses, times, nitr=10):
    """Previous integration step with the default, no-save checkpoint policy."""
    ki = symp._compute_ki(masses)
    dtarr = jnp.diff(times)
    x, v = symp.real_to_mapTO(x, v, ki, masses, dtarr[0])
    x, v = symp.kepler_step(x, v, ki, dtarr[0]*0.5, nitr=nitr)

    def step(xvin, dt):
        x, v = xvin
        x, v = symp.nbody_kicks(x, v, ki, masses, dt)
        xout, vout = symp.kepler_step(x, v, ki, dt, nitr=nitr)
        return [xout, vout], jnp.array([xout, vout])

    _, xv = jax.lax.scan(jax.checkpoint(step, prevent_cse=False), [x, v], dtarr)
    return times[1:]+0.5*dtarr[0], xv


@pytest.fixture(scope="module", params=("kep51", "realistic8_reduced"))
def problem(request):
    if request.param == "kep51":
        jttv, _, _, params = read_testdata_tc()
        keys = ("period", "ecosw", "esinw", "cosi", "lnode", "tic", "pmass")
    else:
        path = Path(__file__).with_name("data")/"selective_checkpoint_realistic8.npz"
        with np.load(path) as data:
            keys = ("period", "ecosw", "esinw", "tic", "lnpmass")
            params = {key: jnp.asarray(data[key]) for key in keys}
            jttv = JaxTTV(float(data["t_start"]), float(data["t_end"]), float(data["dt"]),
                          [np.array([t]) for t in data["tcobs"]], data["p_init"],
                          errorobs=[np.array([s]) for s in data["errorobs"]], print_info=False)
    nplanet = len(params["period"])
    theta = jnp.concatenate([jnp.asarray(params[key]) for key in keys])

    def unpack(theta):
        model = {key: theta[i*nplanet:(i+1)*nplanet] for i, key in enumerate(keys)}
        if "lnpmass" in model:
            model["pmass"] = jnp.exp(model["lnpmass"])
        return model

    def transit_times(theta, integrate):
        x, v, masses = initialize_jacobi_xv(unpack(theta), jttv.t_start)
        times, xv = integrate(x, v, masses, jttv.times, nitr=jttv.nitr_kepler)
        return jttv._compute_transit_times(jttv.pidx.astype(int)-1,
                                         jttv.tcobs_flatten, times, xv, masses)

    def residual(theta, integrate):
        return (jttv.tcobs_flatten-transit_times(theta, integrate))/jttv.errorobs_flatten

    return jttv, theta, unpack, transit_times, residual


def test_forward_integration_unchanged(problem):
    jttv, theta, unpack, _, _ = problem
    inputs = (*initialize_jacobi_xv(unpack(theta), jttv.t_start), jttv.times)
    expected = jax.jit(_integrate_baseline)(*inputs)
    actual = jax.jit(symp.integrate_xv)(*inputs)
    assert actual[1].shape == (len(jttv.times)-1, 2, len(unpack(theta)["period"]), 3)
    for got, reference in zip(actual, expected):
        npt.assert_allclose(got, reference, rtol=1e-12, atol=1e-12)


def test_transits_residual_loss_and_gradient_unchanged(problem):
    jttv, theta, unpack, _, residual = problem

    def loss(theta, integrate):
        return jnp.sum(residual(theta, integrate)**2)

    results = []
    for integrate in (_integrate_baseline, symp.integrate_xv):
        function = jax.jit(jax.value_and_grad(lambda t: loss(t, integrate)))
        results.append(function(theta))
    expected_loss, expected_gradient = results[0]
    actual_loss, gradient = results[1]
    expected_r = jax.jit(lambda t: residual(t, _integrate_baseline))(theta)
    r = jax.jit(lambda t: residual(t, symp.integrate_xv))(theta)
    npt.assert_allclose(r, expected_r, rtol=1e-10, atol=1e-7)
    npt.assert_allclose(actual_loss, expected_loss, rtol=1e-10, atol=1e-7)
    npt.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-7)
    # Exercise the public production method too; the fast finder is unchanged.
    tc = jttv.get_transit_times_obs(unpack(theta))[0]
    expected_tc = jttv.tcobs_flatten-expected_r*jttv.errorobs_flatten
    npt.assert_allclose(tc, expected_tc, rtol=0, atol=1e-8)


def test_gradient_with_residual_aux_unchanged_in_norm(problem):
    _, theta, _, _, residual = problem
    outputs = []
    for integrate in (_integrate_baseline, symp.integrate_xv):
        def loss(t):
            r = residual(t, integrate)
            return jnp.sum(r**2), r
        outputs.append(jax.jit(jax.value_and_grad(loss, has_aux=True))(theta))
    (expected_loss, expected_r), expected_gradient = outputs[0]
    (loss, r), gradient = outputs[1]
    npt.assert_allclose(r, expected_r, rtol=1e-10, atol=1e-7)
    npt.assert_allclose(loss, expected_loss, rtol=1e-10, atol=1e-7)
    # Additional primal outputs can change fusion/rounding in the baseline.
    # Use a vector bound for this context; the scalar path above also checks
    # every component, including weak gradients, at the same relative tolerance.
    error = np.linalg.norm(np.asarray(gradient-expected_gradient))
    assert error <= 1e-7+1e-8*np.linalg.norm(np.asarray(expected_gradient))


def test_dense_jacobian_unchanged(problem):
    _, theta, _, _, residual = problem
    jacobians = [jax.jit(jax.jacfwd(lambda t: residual(t, integrate)))(theta)
                 for integrate in (_integrate_baseline, symp.integrate_xv)]
    npt.assert_allclose(jacobians[1], jacobians[0], rtol=1e-10, atol=1e-8)


def _equations(jaxpr):
    if hasattr(jaxpr, "jaxpr"):
        jaxpr = jaxpr.jaxpr
    for equation in jaxpr.eqns:
        yield equation
        for value in equation.params.values():
            for child in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(child, "eqns") or hasattr(child, "jaxpr"):
                    yield from _equations(child)


def test_only_dE_saved_and_backward_kepler_loop_removed(problem):
    jttv, theta, unpack, _, residual = problem
    length, nplanet = len(jttv.times)-1, len(unpack(theta)["period"])
    outputs = []
    for integrate, expected_reverse_whiles in ((_integrate_baseline, 1), (symp.integrate_xv, 0)):
        loss = lambda t: jnp.sum(residual(t, integrate)**2)
        closed = jax.make_jaxpr(jax.value_and_grad(loss))(theta)
        scans = [e for e in _equations(closed)
                 if e.primitive.name == "scan" and e.params["length"] == length]
        assert len(scans) == 2
        forward = next(e for e in scans if not e.params["reverse"])
        reverse = next(e for e in scans if e.params["reverse"])
        assert sum(e.primitive.name == "while" for e in _equations(forward.params["jaxpr"])) == 1
        remats = [e for e in _equations(reverse.params["jaxpr"]) if e.primitive.name.startswith("remat")]
        assert len(remats) == 1
        assert sum(e.primitive.name == "while" for e in _equations(remats[0].params["jaxpr"])) == expected_reverse_whiles
        outputs.append(Counter((v.aval.shape, str(v.aval.dtype)) for v in forward.outvars))
    assert outputs[1]-outputs[0] == Counter({((length, nplanet), "float64"): 1})
    assert not outputs[0]-outputs[1]
