import numpy as np
import jax.numpy as jnp

from jnkepler.tests import read_testdata_tc
from jnkepler.jaxttv.findtransit import (
    find_transit_times_all,
    find_transit_times_fast,
    find_transit_params_all,
    find_transit_params_fast,
)
from jnkepler.jaxttv.jaxttv import initialize_jacobi_xv
from jnkepler.jaxttv.symplectic import integrate_xv


def _get_test_integration():
    """Return a representative integrated system for transit-finder tests."""
    jttv, _, _, pdic = read_testdata_tc()
    xjac0, vjac0, masses = initialize_jacobi_xv(pdic, jttv.t_start)
    times, xvjac = integrate_xv(
        xjac0, vjac0, masses, jttv.times, nitr=jttv.nitr_kepler
    )
    pidxarr = jttv.pidx.astype(int) - 1
    tcobsarr = jttv.tcobs_flatten
    return jttv, pidxarr, tcobsarr, times, xvjac, masses


def test_find_transit_times_fast_matches_newton():
    jttv, pidxarr, tcobsarr, times, xvjac, masses = _get_test_integration()

    tc_fast = find_transit_times_fast(pidxarr, tcobsarr, times, xvjac, masses)
    tc_newton = find_transit_times_all(
        pidxarr, tcobsarr, times, xvjac, masses, nitr=jttv.nitr_transit
    )

    assert tc_fast.shape == tc_newton.shape
    assert np.all(np.isfinite(np.asarray(tc_fast)))
    assert np.allclose(tc_fast, tc_newton, rtol=0.0, atol=1e-6)


def test_find_transit_params_fast_matches_newton():
    jttv, pidxarr, tcobsarr, times, xvjac, masses = _get_test_integration()

    tc_fast, (xcm_fast, vcm_fast, dt_fast) = find_transit_params_fast(
        pidxarr, tcobsarr, times, xvjac, masses
    )
    tc_newton, (xcm_newton, vcm_newton, dt_newton) = find_transit_params_all(
        pidxarr, tcobsarr, times, xvjac, masses, nitr=jttv.nitr_transit
    )

    assert tc_fast.shape == tc_newton.shape
    assert xcm_fast.shape == xcm_newton.shape
    assert vcm_fast.shape == vcm_newton.shape
    assert dt_fast.shape == dt_newton.shape

    assert np.all(np.isfinite(np.asarray(tc_fast)))
    assert np.all(np.isfinite(np.asarray(xcm_fast)))
    assert np.all(np.isfinite(np.asarray(vcm_fast)))
    assert np.all(np.isfinite(np.asarray(dt_fast)))

    assert np.allclose(tc_fast, tc_newton, rtol=0.0, atol=1e-6)
    assert np.allclose(xcm_fast, xcm_newton, rtol=0.0, atol=1e-6)
    assert np.allclose(vcm_fast, vcm_newton, rtol=0.0, atol=1e-6)

    # Returned corrections should be small and the selected planets should be
    # close to the transit center (x · v ≈ 0 in the sky plane).
    dt_step = float(np.diff(np.asarray(times))[0])
    assert np.all(np.abs(np.asarray(dt_fast)) < 0.1 * dt_step)
