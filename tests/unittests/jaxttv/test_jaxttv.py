import numpy as np
import jax.numpy as jnp
import importlib_resources
from importlib.util import find_spec
import pickle
from jnkepler.tests import read_testdata_tc, read_testdata_4planet
from jnkepler.jaxttv.ttvfastutils import params_for_ttvfast, get_ttvfast_model_rv, get_ttvfast_model
from jnkepler.jaxttv.utils import em_to_dict, params_to_elements, elements_to_pdic, findidx_map

path = importlib_resources.files('jnkepler').joinpath('data')


def test_jaxttv():
    jttv, _, _, _ = read_testdata_tc()
    with open(path/"JaxTTVobject.pkl", "rb") as f:
        jttv_ref = pickle.load(f)
    for attr, value in vars(jttv_ref).items():
        assert hasattr(
            jttv, attr), f"JaxTTV object is missing attribute {attr}"
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            assert np.allclose(getattr(
                jttv, attr), value), f"Mismatch in {attr}: {getattr(jttv, attr)} != {value}"
        elif isinstance(value, list):
            for x, y in zip(getattr(jttv, attr), value):
                assert np.array_equal(
                    x, y), f"Mismatch in {attr}: {getattr(jttv, attr)} != {value}"
        else:
            assert getattr(
                jttv, attr) == value, f"Mismatch in {attr}: {getattr(jttv, attr)} != {value}"


def test_get_transit_times_all():
    jttv, _, _, _ = read_testdata_tc()

    elements = np.loadtxt(path/"tcbug_elements.txt")
    masses = np.loadtxt(path/"tcbug_masses.txt")
    pdic = em_to_dict(elements, masses)

    tc_jttv, _, _ = jttv.get_transit_times_all(pdic)
    if find_spec("ttvfast") is None:
        tcs_ttvfast = np.loadtxt(path/"tcs_ttvfast_for_test-all.txt")
    else:
        tcs_ttvfast = get_ttvfast_times(jttv, pdic, jttv.nplanet, obs=False)

    assert np.allclose(tc_jttv, tcs_ttvfast, rtol=0., atol=1e-5)


def test_check_timing_precision():
    jttv, _, _, pdic = read_testdata_tc()
    tc, tc2 = jttv.check_timing_precision(pdic)

    assert np.isclose(np.max(np.abs(tc-tc2)), 6.10e-6)


def test_get_transit_times_obs_nontransiting():
    jttv, params, params2 = read_testdata_4planet()
    params_, _ = np.array(params.iloc[0])[:-1], np.array(params.iloc[0])[-1]
    params2_, _ = np.array(params2.iloc[1])[:-1], np.array(params2.iloc[1])[-1]

    pdic = elements_to_pdic(
        *params_to_elements(params_, 4), force_coplanar=False)
    pdic['pmass'] = pdic['mass']
    transit_idx = jnp.array([0., 1, 2])
    tc = jttv.get_transit_times_obs(pdic, transit_idx)[0]
    chi2 = np.sum(((tc - jttv.tcobs_flatten) / jttv.errorobs_flatten)**2)
    if find_spec("ttvfast") is None:
        tc_tf = np.loadtxt(path/"tcs_ttvfast_for_test-obs-nontransiting.txt")
    else:
        tc_tf = get_ttvfast_times(jttv, pdic, 4)
    chi2_ref = np.sum(
        ((tc_tf - jttv.tcobs_flatten) / jttv.errorobs_flatten)**2)
    assert np.isclose(chi2, chi2_ref, rtol=1e-2)

    # insufficient precision for this parameter set?
    pdic = elements_to_pdic(
        *params_to_elements(params2_, 4), force_coplanar=False)
    pdic['pmass'] = pdic['mass']
    transit_idx = jnp.array([0, 1, 3])
    tc = jttv.get_transit_times_obs(pdic, transit_idx)[0]
    chi2 = np.sum(((tc - jttv.tcobs_flatten) / jttv.errorobs_flatten)**2)
    if find_spec("ttvfast") is None:
        tc_tf = np.loadtxt(path/"tcs_ttvfast_for_test-obs-nontransiting2.txt")
    else:
        tc_tf = get_ttvfast_times(jttv, pdic, 4)
    chi2_ref = np.sum(
        ((tc_tf - jttv.tcobs_flatten) / jttv.errorobs_flatten)**2)

    assert np.isclose(chi2, chi2_ref, rtol=1e-2)


def test_get_transit_times_all_nontransiting():
    jttv, params, _ = read_testdata_4planet()
    params_ = np.array(params.iloc[0])[:-1]
    pdic = elements_to_pdic(
        *params_to_elements(params_, 4), force_coplanar=False)
    pdic['pmass'] = pdic['mass']

    transit_orbit_idx = np.array([0, 1, 2])

    tc_jttv, _, oidx = jttv.get_transit_times_all(
        pdic, transit_orbit_idx=transit_orbit_idx)
    if find_spec("ttvfast") is None:
        tcs_ttvfast = np.loadtxt(path/"tcs_ttvfast_for_test-nontransiting.txt")
    else:
        tcs_ttvfast = get_ttvfast_times(
            jttv, pdic, jttv.nplanet+1, obs=False, skip_planet_idx=[3])

    assert np.allclose(tc_jttv, tcs_ttvfast, rtol=0., atol=2e-5)


def test_get_transit_times_and_rvs_obs():
    jttv, _, _, _ = read_testdata_tc()

    elements = np.loadtxt(path/"tcbug_elements.txt")
    masses = np.loadtxt(path/"tcbug_masses.txt")
    pdic = em_to_dict(elements, masses)

    times_rv = np.linspace(jttv.t_start+5, jttv.t_end-5, 10000)
    if find_spec("ttvfast") is None:
        tc_ttvfast = np.loadtxt(path/"tcs_ttvfast_for_test-ttandrv.txt")
        rv_ttvfast = np.loadtxt(path/"rvs_ttvfast_for_test-ttandrv.txt")
    else:
        tc_ttvfast, rv_ttvfast = get_ttvfast_times_and_rvs(
            jttv, pdic, jttv.nplanet, times_rv)

    tc_jttv, rv_jttv, _ = jttv.get_transit_times_and_rvs_obs(pdic, times_rv)

    assert np.allclose(tc_jttv, tc_ttvfast, rtol=0, atol=1e-5)
    assert np.allclose(rv_jttv, rv_ttvfast, rtol=0, atol=5e-3)


def get_ttvfast_times(jttv, pdic, nplanet, obs=True, **kwargs):
    p = {}

    for key in pdic.keys():
        p[key] = np.array([pdic[key]])

    pdic_ttvfast = params_for_ttvfast(p, jttv.t_start, nplanet)
    _, tcs_ttvfast = get_ttvfast_model(
        pdic_ttvfast.iloc[0], nplanet, jttv.t_start, jttv.dt, jttv.t_end, **kwargs)
    tcs_ttvfast = np.hstack(tcs_ttvfast)

    if not obs:
        return tcs_ttvfast

    idx_for_tf = findidx_map(tcs_ttvfast, jttv.tcobs_flatten)
    tc_tf = tcs_ttvfast[idx_for_tf]

    return tc_tf


def get_ttvfast_times_and_rvs(jttv, pdic, nplanet, times_rv, obs=True, **kwargs):
    p = {}

    for key in pdic.keys():
        p[key] = np.array([pdic[key]])

    pdic_ttvfast = params_for_ttvfast(p, jttv.t_start, nplanet)
    _, tcs_ttvfast, rvs = get_ttvfast_model_rv(
        pdic_ttvfast.iloc[0], nplanet, jttv.t_start, jttv.dt, jttv.t_end, times_rv,  **kwargs)
    tcs_ttvfast = np.hstack(tcs_ttvfast)

    if obs:
        idx_for_tf = findidx_map(tcs_ttvfast, jttv.tcobs_flatten)
        tcs_ttvfast = tcs_ttvfast[idx_for_tf]

    return tcs_ttvfast, rvs


if __name__ == '__main__':
    test_jaxttv()
    test_get_transit_times_all()
    test_check_timing_precision()
    test_get_transit_times_obs_nontransiting()
    test_get_transit_times_all_nontransiting()
    test_get_transit_times_and_rvs_obs()
