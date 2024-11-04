import numpy as np
import jax.numpy as jnp
import importlib_resources, pickle
from jnkepler.tests import read_testdata_tc
from jnkepler.jaxttv.ttvfastutils import params_for_ttvfast, get_ttvfast_model_all
from jnkepler.jaxttv.utils import em_to_dict

path = importlib_resources.files('jnkepler').joinpath('data')

def test_jaxttv():
    jttv, _, _, _ = read_testdata_tc()
    with open(path/"JaxTTVobject.pkl", "rb") as f:
        jttv_ref = pickle.load(f)
    for attr, value in vars(jttv_ref).items():
        assert hasattr(jttv, attr), f"JaxTTV object is missing attribute {attr}"
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            assert np.allclose(getattr(jttv, attr), value), f"Mismatch in {attr}: {getattr(jttv, attr)} != {value}"
        elif isinstance(value, list):
            for x, y in zip(getattr(jttv, attr), value):
                assert np.array_equal(x, y), f"Mismatch in {attr}: {getattr(jttv, attr)} != {value}"
        else:
            assert getattr(jttv, attr) == value, f"Mismatch in {attr}: {getattr(jttv, attr)} != {value}"


def test_get_transit_times_all():
    jttv, _, _, _ = read_testdata_tc()

    elements = np.loadtxt(path/"tcbug_elements.txt")
    masses = np.loadtxt(path/"tcbug_masses.txt")
    pdic = em_to_dict(elements, masses)

    tc_jttv, _, _ = jttv.get_transit_times_all(pdic)

    for key in pdic.keys():
        pdic[key] = np.array([pdic[key]])
    pdic_ttvfast = params_for_ttvfast(pdic, jttv.t_start, jttv.nplanet)
    _, tcs_ttvfast, _, _ = get_ttvfast_model_all(pdic_ttvfast.iloc[0], jttv.nplanet, jttv.t_start, jttv.dt, jttv.t_end)
    tcs_ttvfast = np.hstack(tcs_ttvfast)

    assert np.allclose(tc_jttv, tcs_ttvfast, rtol=0., atol=1e-5)


def test_check_timing_precision():
    jttv, _, _, pdic = read_testdata_tc()
    tc, tc2 = jttv.check_timing_precision(pdic)

    assert np.isclose(np.max(np.abs(tc-tc2)), 6.10e-6)


if __name__ == '__main__':
    test_jaxttv()
    test_get_transit_times_all()
    test_check_timing_precision()
