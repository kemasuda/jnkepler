#%%
import pytest
import glob
import pandas as pd
import numpy as np
from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv.utils import params_to_elements, convert_elements
import pkg_resources

#%%
path = pkg_resources.resource_filename('jnkepler', 'data/')

#%%
def test_convert_elements():
    p_init = [45.155305, 85.31646, 130.17809]
    d = pd.read_csv(path+"kep51_ttv.txt", delim_whitespace=True, header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [np.array(d.tc[d.planum==j+1]) for j in range(3)]
    params_jttv = np.loadtxt(path+"kep51_dt1.0_start155.00_end2950.00_params.txt")
    t_start = 155.
    elements, masses = params_to_elements(params_jttv, len(p_init))
    _, period, ecc, inc, omega, lnode, ma = convert_elements(elements, masses, t_start, WHsplit=True)[0]
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
    test_convert_elements()
