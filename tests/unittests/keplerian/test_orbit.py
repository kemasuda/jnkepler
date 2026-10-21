import numpy as np
import pytest
from jnkepler.keplerian import elements_to_xv, elements_to_xv_scaled, xv_to_elements


@pytest.mark.parametrize("inc", [0., np.pi])
@pytest.mark.parametrize("ecc", [0., 0.2, 0.8])
def test_equatorial_orbits_preserve_orientation(inc, ecc):
    params = dict(period=100., ecc=ecc, inc=inc, omega=0.5,
                  lnode=1., tau=10., mass=1.)
    initial = elements_to_xv(0., params)
    # Exact equatorial input, including retrograde motion (sin(pi) is not
    # exactly zero in floating point).
    x, v = np.array(initial["x"][0]), np.array(initial["v"][0])
    x[2], v[2] = 0., 0.
    recovered = xv_to_elements(x, v, params["mass"], t_ref=0.)
    assert recovered["lnode"] == 0.
    if ecc > 0.:
        expected_omega = params["omega"] + (1. if inc == 0. else -1.)*params["lnode"]
        np.testing.assert_allclose(np.exp(1j*recovered["omega"]),
                                   np.exp(1j*expected_omega), atol=1e-14)
    # Check the physical trajectory, since the individual angles (and tau
    # for circular motion) need not match the original parameterization.
    times = np.array([0., 5., 20.])
    expected = elements_to_xv(times, params)
    actual = elements_to_xv(times, recovered)
    for key in ("x", "v"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=5e-15)


def test_elements_xv_reversible():
    """Check that elements_to_xv and xv_to_elements are (approximately) inverse."""
    t = np.linspace(0., 20., 5)
    params = dict(
        period=100.,
        ecc=0.2,
        inc=0.3,
        omega=0.5,
        lnode=1.0,
        tau=10.0,
        mass=1.0,
    )
    # forward
    out = elements_to_xv(t, params)
    x, v = out["x"], out["v"]

    # shape
    assert x.shape == (len(t), 3)
    assert v.shape == (len(t), 3)

    # backward
    t_idx = 2
    elems = xv_to_elements(x[t_idx], v[t_idx], params["mass"], t_ref=t[t_idx])
    for key in params.keys():
        assert np.allclose(elems[key], params[key])


def test_elements_xv_reversible_vectors():
    """Check that elements_to_xv and xv_to_elements are (approximately) inverse."""
    t = np.linspace(0., 20., 5)
    params = dict(
        period=np.array([100., 200.]),
        ecc=np.array([0.2, 1e-6]),
        inc=np.array([1e-6, 0.5*np.pi]),
        omega=np.array([0.5, 0.1]),
        lnode=np.array([1.0, 0.1]),
        tau=np.array([10.0, 5.]),
        mass=1.0,
    )
    # forward
    out = elements_to_xv(t, params)
    x, v = out["x"], out["v"]

    # shape
    N = len(params['period'])
    assert x.shape == (len(t), N, 3)
    assert v.shape == (len(t), N, 3)

    # backward
    t_idx = 2
    elems = xv_to_elements(x[t_idx], v[t_idx], params["mass"], t_ref=t[t_idx])
    for key in params.keys():
        assert np.allclose(elems[key], params[key])


def test_elements_to_xv_scaled():
    """Check elements_to_xv_scaled."""
    t = 0.
    params = dict(
        period=100.,
        ecc=0.2,
        inc=0.3,
        omega=0.5,
        lnode=1.0,
        tau=10.0,
        mass=1.0,
    )

    out = elements_to_xv(t, params)
    x, v = out["x"], out["v"]

    out_scaled = elements_to_xv_scaled(t, params)
    x_scaled, v_scaled = out_scaled['x'], out_scaled['v']
    elems = xv_to_elements(x, v, params["mass"])
    a = elems['a']

    assert np.allclose(x / a, x_scaled)
    assert np.allclose(v / a, v_scaled)


if __name__ == '__main__':
    test_elements_xv_reversible()
    test_elements_xv_reversible_vectors()
    test_elements_to_xv_scaled()
