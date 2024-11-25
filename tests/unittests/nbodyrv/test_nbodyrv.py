from jnkepler.nbodyrv import NbodyRV
from jnkepler.tests import read_testdata_tc
from jnkepler.jaxttv.utils import em_to_dict
import numpy as np
import importlib_resources

path = importlib_resources.files('jnkepler').joinpath('data')


def test_get_rv():
    jttv, _, _, _ = read_testdata_tc()
    elements = np.loadtxt(path/"tcbug_elements.txt")
    masses = np.loadtxt(path/"tcbug_masses.txt")
    pdic = em_to_dict(elements, masses)

    nrv = NbodyRV(jttv.t_start, jttv.t_end, jttv.dt)
    times_rv = np.linspace(jttv.t_start+5, jttv.t_end-5, 10000)
    rv_nbodyrv = nrv.get_rv(times_rv, pdic)

    rv_ttvfast = np.loadtxt(path/"rvs_ttvfast_for_test-ttandrv.txt")

    assert np.allclose(rv_nbodyrv, rv_ttvfast, rtol=0, atol=5e-3)


if __name__ == '__main__':
    test_get_rv()
