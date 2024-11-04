#%%
import pytest
import numpy as np
import jax.numpy as jnp
import pandas as pd
import importlib_resources
from jnkepler.nbodytransit import *
from jnkepler.jaxttv.utils import params_to_elements, em_to_dict
from jax import config
config.update('jax_enable_x64', True)

#%%
path = importlib_resources.files('jnkepler').joinpath('data')

#%%
def compute_testlc():
    d = pd.read_csv(path/"kep51_ttv_photodtest.txt", sep="\s+", header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [jnp.array(d.tc[d.planum==j+1]) for j in range(3)]
    errorobs = [jnp.array(d.tcerr[d.planum==j+1]) for j in range(3)]
    p_init = [45.155305, 85.31646, 130.17809]
    dt = 1.0
    t_start, t_end = 155., 1500.

    times_lc = jnp.array(pd.read_csv(path+"kep51_lc_photodtest.csv")['time'])
    params = np.loadtxt(path/"kep51_dt1.0_start155.00_end2950.00_params.txt")
    elements, masses = params_to_elements(params, 3)
    par_dict = em_to_dict(elements, masses)
    rstar, u1, u2 = 1., 0.5, 0.2
    prad = jnp.array([0.07, 0.05, 0.1])

    nt = NbodyTransit(t_start, t_end, dt, tcobs, p_init, errorobs=errorobs, print_info=False)
    nt.set_lcobs(times_lc)
    lc, ttv = nt.get_lc(par_dict, rstar, prad, u1, u2)
    np.savetxt(path/"kep51_lc_model.txt", lc)

#%%
#compute_testlc()

#%%
def test_reproduce():
    d = pd.read_csv(path/"kep51_ttv_photodtest.txt", sep="\s+", header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [jnp.array(d.tc[d.planum==j+1]) for j in range(3)]
    errorobs = [jnp.array(d.tcerr[d.planum==j+1]) for j in range(3)]
    p_init = [45.155305, 85.31646, 130.17809]
    dt = 1.0
    t_start, t_end = 155., 1500.

    times_lc = jnp.array(pd.read_csv(path/"kep51_lc_photodtest.csv")['time'])
    params = np.loadtxt(path/"kep51_dt1.0_start155.00_end2950.00_params.txt")
    elements, masses = params_to_elements(params, 3)
    par_dict = em_to_dict(elements, masses)
    rstar, u1, u2 = 1., 0.5, 0.2
    prad = jnp.array([0.07, 0.05, 0.1])

    nt = NbodyTransit(t_start, t_end, dt, tcobs, p_init, errorobs=errorobs, print_info=False)
    nt.set_lcobs(times_lc)
    lc, ttv = nt.get_lc(par_dict, rstar, prad, u1, u2)

    lc_test = np.loadtxt(path/"kep51_lc_model.txt")

    assert lc == pytest.approx(lc_test-1.)
    print ("# max fractional difference:", np.max(np.abs(lc-lc_test)))

#%%
if __name__ == '__main__':
    test_reproduce()
