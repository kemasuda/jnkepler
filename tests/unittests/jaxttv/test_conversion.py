#%%
import pytest
import glob
import pandas as pd
import numpy as np
from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv.utils import params_to_elements, convert_elements, em_to_dict
from jnkepler.tests import read_testdata_tc
import importlib_resources
path = importlib_resources.files('jnkepler').joinpath('data')

#%%
def test_convert_elements():
    jttv, params_jttv, _, pdic = read_testdata_tc()
    p_init = jttv.p_init
    _, period, ecc, inc, omega, lnode, ma = convert_elements(pdic, jttv.t_start, WHsplit=True)[0]
    p = {'period': period, 'ecc': ecc, 'cosi': np.cos(inc), 'omega': omega, 'lnode': lnode, 'ma': ma}

    ptrue = pd.read_csv(path/"kep51_dt1.0_start155.00_end2950.00_pdict_ttvfast.csv")

    assert p['period'] == pytest.approx(np.array(ptrue[['period%d'%i for i in range(len(p_init))]])[0])
    assert p['ecc'] == pytest.approx(np.array(ptrue[['e%d'%i for i in range(len(p_init))]])[0])
    assert np.rad2deg(np.arccos(p['cosi'])) == pytest.approx(np.array(ptrue[['incl%d'%i for i in range(len(p_init))]])[0])
    assert np.rad2deg(p['lnode']) == pytest.approx(np.array(ptrue[['lnode%d'%i for i in range(len(p_init))]])[0])
    assert np.rad2deg(p['omega']) == pytest.approx(np.array(ptrue[['omega%d'%i for i in range(len(p_init))]])[0])
    assert np.rad2deg(p['ma']) == pytest.approx(np.array(ptrue[['M%d'%i for i in range(len(p_init))]])[0])

#%%
if __name__ == '__main__':
    test_convert_elements()
