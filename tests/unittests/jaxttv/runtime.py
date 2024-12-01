#%%
import pytest
import glob
import pandas as pd
import numpy as np
from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv.utils import params_to_elements, elements_to_pdic
from jax import grad, jit
import jax.numpy as jnp
import pkg_resources
import timeit

#%%
path = pkg_resources.resource_filename('jnkepler', 'data/')

#%%
def test_runtime():
    pltag = 'kep51'
    p_init = [45.155305, 85.31646, 130.17809]
    d = pd.read_csv(path+"%s_ttv.txt"%pltag, sep=r'\s+', header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [np.array(d.tc[d.planum==j+1]) for j in range(3)]
    params_test = np.loadtxt(glob.glob(path+"%s*params.txt"%pltag)[0])
    tc_test = np.array(pd.read_csv(glob.glob(path+"%s*tc.csv"%pltag)[0])).ravel()

    dt = 1.0
    t_start, t_end = 155., 2950.
    jttv = JaxTTV(t_start, t_end, dt, tcobs, p_init, print_info=False)
    elements, masses = params_to_elements(params_test, jttv.nplanet)
    pdic = elements_to_pdic(elements, masses)
    pdic['mass_ratio'] = pdic['mass']
    _, _ = jttv.get_transit_times_obs(pdic)

    names, loop = {**globals(), **locals()}, 200
    result = timeit.timeit('jttv.get_transit_times_obs(pdic)', globals=names, number=loop)
    result_str = "%.2f ms per loop (%d loops)"%(result / loop * 1000, loop)
    print ("# JaxTTV (NR) timeit:", result_str)

    """ interpolation
    jttv = JaxTTV(t_start, t_end, dt)
    jttv.set_tcobs(tcobs, p_init, print_info=False)
    jttv.transit_time_method = 'interpolation'
    _, _ = jttv.get_ttvs(*params_to_elements(params_test, jttv.nplanet))

    names, loop = {**globals(), **locals()}, 200
    result = timeit.timeit('jttv.get_ttvs(*params_to_elements(params_test, jttv.nplanet))', globals=names, number=loop)
    result_str = "%.2f ms per loop (%d loops)"%(result / loop * 1000, loop)
    print ("# JaxTTV (interp) timeit:", result_str)
    """

    func = lambda pdic: jnp.sum(jttv.get_transit_times_obs(pdic)[0])
    gfunc = jit(grad(func, argnums=(0,)))
    _ = gfunc(pdic)
    names = {**globals(), **locals()}
    result = timeit.timeit('gfunc(pdic)', globals=names, number=loop)
    result_str = "%.2f ms per loop (%d loops)"%(result / loop * 1000, loop)
    print ("# JaxTTV (NR) gradient timeit:", result_str)

#%%
if __name__ == '__main__':
    test_runtime()
