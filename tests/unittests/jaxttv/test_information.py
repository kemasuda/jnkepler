import numpy as np
import importlib_resources
from jnkepler.tests import read_testdata_tc
from jnkepler.jaxttv.information import *
from jnkepler.jaxttv.infer import ttv_default_parameter_bounds

path = importlib_resources.files('jnkepler').joinpath('data')


def test_information():
    info_ref = np.loadtxt(path/"info.txt")

    jttv, _, _, pdic = read_testdata_tc()
    sample_keys = ['ecosw', 'esinw', 'pmass', 'period', 'tic']
    info = information(jttv, pdic, sample_keys)

    assert np.allclose(info, info_ref)


def test_information_scale():
    jttv, _, _, pdic = read_testdata_tc()
    sample_keys = ['ecosw', 'esinw', 'pmass', 'period', 'tic']
    param_bounds = ttv_default_parameter_bounds(jttv)

    info = information(jttv, pdic, sample_keys)
    scale_info = scale_information(info, param_bounds, sample_keys)
    scale_info2 = information(jttv, pdic, sample_keys,
                              param_bounds=param_bounds)

    assert np.allclose(scale_info, scale_info2)


"""experimental functions
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
"""

if __name__ == '__main__':
    test_information()
    test_information_scale()
    # test_hessian()
    # test_observed_information()
