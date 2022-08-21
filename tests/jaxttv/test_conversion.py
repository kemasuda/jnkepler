#%%
import pytest
import glob
import pandas as pd
import numpy as np
from jnkepler.jaxttv import JaxTTV, params_to_elements
import pkg_resources

#%%
path = pkg_resources.resource_filename('jnkepler', 'data/')

#%%
def test_conversion():
    p_init = [45.155305, 85.31646, 130.17809]
    d = pd.read_csv(path+"kep51_ttv.txt", delim_whitespace=True, header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [np.array(d.tc[d.planum==j+1]) for j in range(3)]
    params_jttv = np.loadtxt(path+"kep51_dt1.0_start155.00_end2950.00_params.txt")
    dt = 0.5 * 2
    t_start, t_end = 155., 1495 + 1455
    jttv = JaxTTV(t_start, t_end, dt)
    jttv.set_tcobs(tcobs, p_init)
    #p = jttv.get_elements(params_jttv, t_start, wh=True)
    _, period, ecc, inc, omega, lnode, ma = jttv.get_elements(params_jttv, WHsplit=True)
    p = {'period': period, 'ecc': ecc, 'cosi': np.cos(inc), 'omega': omega, 'lnode': lnode, 'ma': ma}

    ptrue = pd.read_csv(path+"kep51_dt1.0_start155.00_end2950.00_pdict_ttvfast.csv")

    assert p['period'] == pytest.approx(np.array(ptrue[['period%d'%i for i in range(len(p_init))]])[0])
    assert p['ecc'] == pytest.approx(np.array(ptrue[['e%d'%i for i in range(len(p_init))]])[0])
    assert np.rad2deg(np.arccos(p['cosi'])) == pytest.approx(np.array(ptrue[['incl%d'%i for i in range(len(p_init))]])[0])
    assert np.rad2deg(p['lnode']) == pytest.approx(np.array(ptrue[['lnode%d'%i for i in range(len(p_init))]])[0])
    assert np.rad2deg(p['omega']) == pytest.approx(np.array(ptrue[['omega%d'%i for i in range(len(p_init))]])[0])
    assert np.rad2deg(p['ma']) == pytest.approx(np.array(ptrue[['M%d'%i for i in range(len(p_init))]])[0])

#%%
if __name__ == '__main__':
    test_coversion()
