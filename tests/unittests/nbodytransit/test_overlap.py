#%%
import pytest
import numpy as np
import jax.numpy as jnp
import pandas as pd
import importlib_resources
from jnkepler.nbodytransit import *
from jax import config
config.update('jax_enable_x64', True)

#%%
path = importlib_resources.files('jnkepler').joinpath('data')

#%%
def test_overlap():
    d = pd.read_csv(path/"kep51_ttv_photodtest.txt", sep='\s+', header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [jnp.array(d.tc[d.planum==j+1]) for j in range(3)]
    errorobs = [jnp.array(d.tcerr[d.planum==j+1]) for j in range(3)]
    p_init = [45.155305, 85.31646, 130.17809]
    dt = 1.0
    t_start, t_end = 155., 1500.

    times_lc = jnp.array(pd.read_csv(path/"kep51_lc_photodtest.csv")['time'])

    elements, masses = np.loadtxt(path/"tcbug_elements.txt"), np.loadtxt(path/"tcbug_masses.txt")
    rstar, u1, u2 = 1., 0.5, 0.2
    prad = jnp.array([0.07, 0.05, 0.1])

    nt = NbodyTransit(t_start, t_end, dt, tcobs, p_init, errorobs=errorobs, print_info=False)
    nt.set_lcobs(times_lc)

    nt_ol = NbodyTransit(t_start, t_end, dt, tcobs, p_init, errorobs=errorobs, print_info=False)
    nt_ol.set_lcobs(times_lc, overlapping_transit=True)

    lc_nool = nt.get_lc(elements, masses, rstar, prad, u1, u2)[0]
    lc_ol = nt_ol.get_lc(elements, masses, rstar, prad, u1, u2)[0]

    assert lc_ol == pytest.approx(lc_nool)

#%%
if __name__ == '__main__':
    test_overlap()
