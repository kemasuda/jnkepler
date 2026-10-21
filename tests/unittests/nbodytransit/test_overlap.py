import pytest
import numpy as np
import jax.numpy as jnp
import pandas as pd
import importlib_resources
from jnkepler.nbodytransit import *
from jax import config
from jnkepler.jaxttv.utils import em_to_dict
config.update('jax_enable_x64', True)

path = importlib_resources.files('jnkepler').joinpath('data')

try:
    import jaxoplanet
    JAXOPLANET_INSTALLED = True
except ImportError:
    JAXOPLANET_INSTALLED = False


@pytest.mark.skipif(not JAXOPLANET_INSTALLED, reason="jaxoplanet is not installed")
def test_overlap():
    d = pd.read_csv(path/"kep51_ttv_photodtest.txt", sep='\s+',
                    header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [jnp.array(d.tc[d.planum == j+1]) for j in range(3)]
    errorobs = [jnp.array(d.tcerr[d.planum == j+1]) for j in range(3)]
    p_init = [45.155305, 85.31646, 130.17809]
    dt = 1.0
    t_start, t_end = 155., 1500.

    times_lc = jnp.array(pd.read_csv(path/"kep51_lc_photodtest.csv")['time'])

    elements, masses = np.loadtxt(
        path/"tcbug_elements.txt"), np.loadtxt(path/"tcbug_masses.txt")
    pdic = em_to_dict(elements, masses)
    pdic["srad"], pdic["u1"], pdic["u2"] = 1., 0.5, 0.2
    pdic["radius_ratio"] = jnp.array([0.07, 0.05, 0.1])
    pdic["smass"] = 1.

    nt = NbodyTransit(t_start, t_end, dt, tcobs, p_init,
                      errorobs=errorobs, print_info=False)
    nt.set_lcobs(times_lc)

    nt_ol = NbodyTransit(t_start, t_end, dt, tcobs, p_init,
                         errorobs=errorobs, print_info=False)
    nt_ol.set_lcobs(times_lc, overlapping_transit=True)

    lc_nool = nt.get_flux(pdic)[0]
    lc_ol = nt_ol.get_flux(pdic)[0]

    assert lc_ol == pytest.approx(lc_nool)


@pytest.mark.skipif(not JAXOPLANET_INSTALLED, reason="jaxoplanet is not installed")
def test_overlap_mode_includes_a_long_nearby_non_simultaneous_transit():
    """True is also required for close transits with unequal durations."""
    from jnkepler.nbodytransit.transit import compute_relative_flux_loss, rsun_au

    params = dict(period=jnp.array([2., 100.]), tic=jnp.array([2.3, 2.]),
                  ecosw=jnp.array([.01, .01]), esinw=jnp.zeros(2),
                  cosi=jnp.zeros(2), pmass=jnp.array([1e-10, 1e-10]),
                  smass=1., srad=1., radius_ratio=jnp.array([.1, .1]), u1=0., u2=0.)
    nt = NbodyTransit(0., 5., .02, [[2.3, 4.3], [2.]], [2., 100.], print_info=False)
    time = jnp.array([2.17])
    nt.set_lcobs(time, overlapping_transit=True, supersample_factor=1, print_info=False)
    flux = nt.get_flux(params)[0]
    tc, x, v = nt.get_xvsky_tc(params)[:3]
    # Independently sum all registered transit events at this one epoch.
    separation = jnp.linalg.norm(x + v*(time[0]-tc)[:, None], axis=1)/rsun_au
    expected = jnp.sum(compute_relative_flux_loss(
        separation, params["radius_ratio"][nt.pidx.astype(int)-1], 0., 0.))
    np.testing.assert_allclose(flux, expected, rtol=0., atol=1e-12)
    assert float(flux[0]) < -.006
    # Contact windows in the same linear sky-plane approximation are disjoint.
    speed = jnp.linalg.norm(v, axis=1)
    half_duration = jnp.sqrt((1.1*rsun_au)**2-jnp.sum(x*x, axis=1))/speed
    assert tc[2] + half_duration[2] < tc[0] - half_duration[0]


if __name__ == '__main__':
    test_overlap()
