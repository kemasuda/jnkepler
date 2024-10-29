import numpy as np
import importlib_resources
from jnkepler.tests import read_testdata_tc
from jnkepler.jaxttv.information import *

path = importlib_resources.files('jnkepler').joinpath('data')

def test_information():
    info_ref = np.loadtxt(path/"info.txt")
    
    jttv, _, _, pdic = read_testdata_tc()
    sample_keys = ['ecosw', 'esinw', 'pmass', 'period', 'tic']
    info = information(jttv, pdic, sample_keys)

    assert np.allclose(info, info_ref)


def test_hessian():
    hess_ref = np.loadtxt(path/"hessian.txt")
    
    jttv, _, _, pdic = read_testdata_tc()
    hess = hessian(jttv, pdic)

    assert np.allclose(hess, hess_ref)


def test_observed_information():
    hess_ref = np.loadtxt(path/"hessian.txt")
    
    jttv, _, _, pdic = read_testdata_tc()
    sample_keys = ['ecosw', 'esinw', 'pmass', 'period', 'tic']
    hess = observed_information(jttv, pdic, sample_keys)

    assert np.allclose(hess, hess_ref)


if __name__ == '__main__':
    test_information()
    test_hessian()
    test_observed_information()
