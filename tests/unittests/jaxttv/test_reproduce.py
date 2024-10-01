#%%
import pytest
import glob
import pandas as pd
import numpy as np
from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv.utils import params_to_elements
from jax import grad, jit
import jax.numpy as jnp
import pkg_resources

#%%
path = pkg_resources.resource_filename('jnkepler', 'data/')

#%%
def test_reproduce():
    pltag = 'kep51'
    p_init = [45.155305, 85.31646, 130.17809]
    d = pd.read_csv(path+"%s_ttv.txt"%pltag, delim_whitespace=True, header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [np.array(d.tc[d.planum==j+1]) for j in range(3)]
    params_test = np.loadtxt(glob.glob(path+"%s*params.txt"%pltag)[0])
    tc_test = np.array(pd.read_csv(glob.glob(path+"%s*tc.csv"%pltag)[0])).ravel()

    dt = 1.0
    t_start, t_end = 155., 2950.

    # transit times from newton-raphson algorithm
    jttv = JaxTTV(t_start, t_end, dt)
    jttv.set_tcobs(tcobs, p_init, print_info=False)
    elements, masses = params_to_elements(params_test, jttv.nplanet)
    tc, de = jttv.get_ttvs(elements, masses)
    maxdiff_day = np.max(np.abs(tc - tc_test))
    assert maxdiff_day < 1e-6
    print ("# JaxTTV (NR) max difference (sec):", maxdiff_day*86400)

    # transit times from interpolation
    jttv = JaxTTV(t_start, t_end, dt)
    jttv.set_tcobs(tcobs, p_init, print_info=False)
    jttv.transit_time_method = 'interpolation'
    tc_interp, _ = jttv.get_ttvs(elements, masses)
    maxdiff_interp_day = np.max(np.abs(tc_interp - tc_test))
    assert maxdiff_interp_day < 1e-6
    print ("# JaxTTV (interp) max difference (sec):", maxdiff_interp_day*86400)

    # energy conservation
    de_true = float(pd.read_csv(glob.glob(path+"%s*de.csv"%pltag)[0]).de)
    assert de == pytest.approx(de_true)
    print ("# true de: %.3e, computed de: %.3e"%(de_true, de))

    # test grad (test data need to be modified)
    jttv = JaxTTV(t_start, t_end, dt)
    jttv.set_tcobs(tcobs, p_init, print_info=False)
    grad_test = np.loadtxt(glob.glob(path+"%s*grad.txt"%pltag)[0])
    func = lambda elements, masses: jnp.sum(jttv.get_ttvs(elements, masses)[0])
    gfunc = jit(grad(func, argnums=(0,)))
    gradval = gfunc(elements, masses)
    fracdiff_grad = (gradval - grad_test) / grad_test
    maxdiff_grad = np.max(np.abs(fracdiff_grad))
    assert maxdiff_grad < 1e-4
    print ("# fractional difference in grad(elements):", fracdiff_grad)

    de_true = float(pd.read_csv(glob.glob(path+"%s*de.csv"%pltag)[0]).de)
    assert de == pytest.approx(de_true)

#%%
if __name__ == '__main__':
    test_reproduce()
