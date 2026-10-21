import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jnkepler.nbodytransit import NbodyTransit

pytest.importorskip("jaxoplanet")


@pytest.fixture(scope="module")
def model():
    nt = NbodyTransit(0., 25., .5, [[2., 12., 22.]], [10.], print_info=False)
    nt.set_lcobs(jnp.array([1.99, 2., 2.01]), supersample_factor=1, print_info=False)
    return nt


def parameters():
    return dict(period=jnp.array([10.]), tic=jnp.array([2.]),
                ecc=jnp.array([.1]), omega=jnp.array([.5]),
                pmass=jnp.array([1e-5]), smass=1., srad=1.,
                b=jnp.array([.3]), radius_ratio=jnp.array([.1]), u1=.3, u2=.2)


@pytest.mark.parametrize("stellar", ["mass_radius", "density"])
@pytest.mark.parametrize("eccentricity", ["polar", "cartesian", "both"])
def test_impact_parameter_supports_orbital_and_stellar_formats(model, stellar, eccentricity):
    reference = parameters()
    reference["ecosw"] = reference["ecc"]*jnp.cos(reference["omega"])
    reference["esinw"] = reference["ecc"]*jnp.sin(reference["omega"])
    reference.pop("ecc")
    reference.pop("omega")
    expected = model.get_flux(reference)
    params = parameters() if eccentricity == "polar" else dict(reference)
    if eccentricity == "both":
        # Cartesian eccentricity takes precedence, as in initialize_jacobi_xv.
        params.update(ecc=jnp.array([.8]), omega=jnp.array([1.]))
    if stellar == "density":
        params.pop("smass")
        params.pop("srad")
        params["sdens"] = 1.
    original_keys = set(params)
    actual = model.get_flux(params)
    for got, wanted in zip(actual, expected):
        np.testing.assert_allclose(got, wanted, rtol=0., atol=1e-12)
    assert set(params) == original_keys


def test_polar_eccentricity_flux_derivative(model):
    def flux(ecc):
        params = parameters()
        params["ecc"] = jnp.array([ecc])
        return model.get_flux(params)[0]

    derivative = jax.jit(jax.jacfwd(flux))(.1)
    delta = 1e-5
    finite_difference = (flux(.1+delta) - flux(.1-delta))/(2*delta)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5, atol=1e-9)
