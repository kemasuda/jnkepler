import numpy as np
#import pkg_resources
import importlib_resources
from jnkepler.tests import read_testdata_tc
from jnkepler.jaxttv.information import *

#path = pkg_resources.resource_filename('jnkepler', 'data/')
path = importlib_resources.files('jnkepler').joinpath('data')

def test_information():
    info_ref = np.loadtxt(path/"info.txt")
    
    jttv, _, _, pdic = read_testdata_tc()
    info = information(jttv, pdic)

    assert np.allclose(info, info_ref)


def test_hessian():
    hess_ref = np.loadtxt(path/"hessian.txt")
    
    jttv, _, _, pdic = read_testdata_tc()
    hess = hessian(jttv, pdic)

    assert np.allclose(hess, hess_ref)


if __name__ == '__main__':
    test_information()
    test_hessian()