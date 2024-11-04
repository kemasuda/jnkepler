#%%
import pytest
import glob
import pandas as pd
import numpy as np
from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv.utils import params_to_elements
from jnkepler.tests import read_testdata_tc
from jax import grad, jit
import jax.numpy as jnp
import pkg_resources

#%%
path = pkg_resources.resource_filename('jnkepler', 'data/')

#%%
def test_reproduce():
    pltag = 'kep51'
    jttv, params_test, tc_test, pdic = read_testdata_tc()

    tc, de = jttv.get_transit_times_obs(pdic)
    maxdiff_day = np.max(np.abs(tc - tc_test))
    assert maxdiff_day < 1e-6
    print ("# JaxTTV (NR) max difference (sec):", maxdiff_day*86400)

    # transit times from interpolation
    jttv, _, _, pdic = read_testdata_tc()
    jttv.transit_time_method = 'interpolation'
    tc_interp, _ = jttv.get_transit_times_obs(pdic)
    maxdiff_interp_day = np.max(np.abs(tc_interp - tc_test))
    assert maxdiff_interp_day < 1e-6
    print ("# JaxTTV (interp) max difference (sec):", maxdiff_interp_day*86400)

    # energy conservation
    de_true = float(pd.read_csv(glob.glob(path+"%s*de.csv"%pltag)[0]).de.iloc[0])
    assert de == pytest.approx(de_true)
    print ("# true de: %.3e, computed de: %.3e"%(de_true, de))

    # test grad (test data need to be modified)
    '''
    jttv, _, _, _ = read_testdata_tc()
    grad_test = np.loadtxt(glob.glob(path+"%s*grad.txt"%pltag)[0])
    func = lambda elements, masses: jnp.sum(jttv.get_transit_times_obs(pdic)[0])
    gfunc = jit(grad(func, argnums=(0,)))
    gradval = gfunc(pdic)
    fracdiff_grad = (gradval - grad_test) / grad_test
    maxdiff_grad = np.max(np.abs(fracdiff_grad))
    assert maxdiff_grad < 1e-4
    print ("# fractional difference in grad(elements):", fracdiff_grad)
    '''

    de_true = float(pd.read_csv(glob.glob(path+"%s*de.csv"%pltag)[0]).de.iloc[0])
    assert de == pytest.approx(de_true)

#%%
if __name__ == '__main__':
    test_reproduce()
