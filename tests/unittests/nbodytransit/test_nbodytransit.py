import pytest
import numpy as np
import jax.numpy as jnp
import pandas as pd
import importlib_resources
from jnkepler.nbodytransit import *
from jnkepler.tests import read_testdata_tc
from jnkepler.jaxttv.utils import params_to_elements, em_to_dict
from jax import config
config.update('jax_enable_x64', True)

path = importlib_resources.files('jnkepler').joinpath('data')

try:
    import jaxoplanet
    JAXOPLANET_INSTALLED = True
except ImportError:
    JAXOPLANET_INSTALLED = False

"""
def compute_testlc():
    d = pd.read_csv(path/"kep51_ttv_photodtest.txt", sep="\s+",
                    header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [jnp.array(d.tc[d.planum == j+1]) for j in range(3)]
    errorobs = [jnp.array(d.tcerr[d.planum == j+1]) for j in range(3)]
    p_init = [45.155305, 85.31646, 130.17809]
    dt = 1.0
    t_start, t_end = 155., 1500.

    times_lc = jnp.array(pd.read_csv(path+"kep51_lc_photodtest.csv")['time'])
    params = np.loadtxt(path/"kep51_dt1.0_start155.00_end2950.00_params.txt")
    elements, masses = params_to_elements(params, 3)
    par_dict = em_to_dict(elements, masses)
    rstar, u1, u2 = 1., 0.5, 0.2
    prad = jnp.array([0.07, 0.05, 0.1])

    nt = NbodyTransit(t_start, t_end, dt, tcobs, p_init,
                      errorobs=errorobs, print_info=False)
    nt.set_lcobs(times_lc)
    lc, ttv = nt.get_flux(par_dict, rstar, prad, u1, u2)
    np.savetxt(path/"kep51_lc_model.txt", lc)
"""


def test_get_xvsky_tc():
    jttv, _, _, pdic = read_testdata_tc()
    nt = NbodyTransit(jttv.t_start, jttv.t_end, jttv.dt,
                      jttv.tcobs, jttv.p_init, print_info=False)

    elements = np.loadtxt(path/"tcbug_elements.txt")
    masses = np.loadtxt(path/"tcbug_masses.txt")
    par_dict = em_to_dict(elements, masses)
    par_dict["srad"], par_dict["u1"], par_dict["u2"] = 1., 0.5, 0.2
    par_dict["radius_ratio"] = jnp.array([0.07, 0.05, 0.1])
    par_dict["smass"] = 1.

    tc_jttv = jttv.get_transit_times_obs(par_dict)[0]
    tc = nt.get_xvsky_tc(par_dict)[0]

    assert np.allclose(tc_jttv, tc)


@pytest.mark.skipif(not JAXOPLANET_INSTALLED, reason="jaxoplanet is not installed")
def test_get_flux():
    d = pd.read_csv(path/"kep51_ttv_photodtest.txt", sep="\s+",
                    header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [jnp.array(d.tc[d.planum == j+1]) for j in range(3)]
    errorobs = [jnp.array(d.tcerr[d.planum == j+1]) for j in range(3)]
    p_init = [45.155305, 85.31646, 130.17809]
    dt = 1.0
    t_start, t_end = 155., 1500.

    times_lc = jnp.array(pd.read_csv(path/"kep51_lc_photodtest.csv")['time'])
    params = np.loadtxt(path/"kep51_dt1.0_start155.00_end2950.00_params.txt")
    elements, masses = params_to_elements(params, 3)
    par_dict = em_to_dict(elements, masses)
    par_dict["srad"], par_dict["u1"], par_dict["u2"] = 1., 0.5, 0.2
    par_dict["radius_ratio"] = jnp.array([0.07, 0.05, 0.1])
    par_dict["smass"] = 1.
    print(par_dict)

    nt = NbodyTransit(t_start, t_end, dt, tcobs, p_init,
                      errorobs=errorobs, print_info=False)
    nt.set_lcobs(times_lc)
    lc, ttv = nt.get_flux(par_dict)

    lc_test = np.loadtxt(path/"kep51_lc_model.txt")

    assert lc == pytest.approx(lc_test-1.)
    print("# max fractional difference:", np.max(np.abs(lc-lc_test)))


@pytest.mark.skipif(not JAXOPLANET_INSTALLED, reason="jaxoplanet is not installed")
def test_get_flux_and_rv():
    jttv, _, _, _ = read_testdata_tc()
    elements = np.loadtxt(path/"tcbug_elements.txt")
    masses = np.loadtxt(path/"tcbug_masses.txt")
    par_dict = em_to_dict(elements, masses)
    par_dict["srad"], par_dict["u1"], par_dict["u2"] = 1., 0.5, 0.2
    par_dict["radius_ratio"] = jnp.array([0.07, 0.05, 0.1])
    par_dict["smass"] = 1.
    times_lc = jnp.array(pd.read_csv(path/"kep51_lc_photodtest.csv")['time'])
    times_rv = np.linspace(jttv.t_start+5, jttv.t_end-5, 10000)

    nt = NbodyTransit(jttv.t_start, jttv.t_end, jttv.dt, jttv.tcobs, jttv.p_init,
                      errorobs=jttv.errorobs, print_info=False)
    nt.set_lcobs(times_lc)
    _, _, rv = nt.get_flux_and_rv(par_dict, times_rv)

    rv_test = np.loadtxt(path/"rvs_ttvfast_for_test-ttandrv.txt")

    assert np.allclose(rv, rv_test, rtol=0, atol=5e-3)


if __name__ == '__main__':
    # compute_testlc()
    test_get_xvsky_tc()
    test_get_flux()
    test_get_flux_and_rv()
