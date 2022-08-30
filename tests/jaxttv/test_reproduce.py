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
    params_jttv = np.loadtxt(glob.glob(path+"%s*params.txt"%pltag)[0])
    tc_jttv = np.array(pd.read_csv(glob.glob(path+"%s*tc.csv"%pltag)[0])).ravel()

    dt = 1.0
    t_start, t_end = 155., 1495 + 1455
    jttv = JaxTTV(t_start, t_end, dt)
    jttv.set_tcobs(tcobs, p_init, print_info=False)
    elements, masses = params_to_elements(params_jttv, jttv.nplanet)
    tc, de = jttv.get_ttvs(elements, masses)
    assert np.max(np.abs(tc - tc_jttv)) < 1e-10 * 1e4
    print ("JaxTTV (NR) max difference (sec):", np.max(np.abs(tc - tc_jttv))*86400)

    import timeit
    names, loop = {**globals(), **locals()}, 200
    result = timeit.timeit('jttv.get_ttvs(*params_to_elements(params_jttv, jttv.nplanet))', globals=names, number=loop)
    result_str = "%.2f ms per loop (%d loops)"%(result / loop * 1000, loop)
    print ("JaxTTV (NR) timeit:", result_str)
    print ()

    jttv = JaxTTV(t_start, t_end, dt)
    jttv.set_tcobs(tcobs, p_init, print_info=False)
    jttv.transit_time_method = 'interpolation'
    tc_interp, de = jttv.get_ttvs(elements, masses)
    assert np.max(np.abs(tc_interp - tc_jttv)) < 1e-6
    print ("JaxTTV (interp) max difference (sec):", np.max(np.abs(tc_interp - tc_jttv))*86400)

    import timeit
    names, loop = {**globals(), **locals()}, 200
    result = timeit.timeit('jttv.get_ttvs(*params_to_elements(params_jttv, jttv.nplanet))', globals=names, number=loop)
    result_str = "%.2f ms per loop (%d loops)"%(result / loop * 1000, loop)
    print ("JaxTTV (interp) timeit:", result_str)
    print ()

    de_true = float(pd.read_csv(glob.glob(path+"%s*de.csv"%pltag)[0]).de)
    assert de == pytest.approx(de_true)

    """
    jttv = JaxTTV(t_start, t_end, dt)
    jttv.set_tcobs(tcobs, p_init, print_info=False)
    grad_jttv = np.loadtxt(glob.glob(path+"%s*grad.txt"%pltag)[0])
    func = lambda elements, masses: jnp.sum(jttv.get_ttvs(elements, masses)[0])
    gfunc = jit(grad(func))
    try:
        assert gfunc(elements, masses) == pytest.approx(grad_jttv)
    except:
        print (gfunc(elements, masses))
        print (grad_jttv)
    """

#%%
if __name__ == '__main__':
    test_reproduce()
