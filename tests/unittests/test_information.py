# autopep8: off
import numpy as np
import importlib_resources
from jnkepler.tests import read_testdata_tc
from jnkepler.jaxttv.information import *
from jnkepler.jaxttv.infer import ttv_default_parameter_bounds
from jnkepler.information import information_from_model_independent_normal
# autopep8: on


path = importlib_resources.files('jnkepler').joinpath('data')


def test_information_with_parameter_dependent_support():
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model():
        lower = numpyro.sample("lower", dist.Uniform(0., 1.))
        x = numpyro.sample("x", dist.Uniform(lower, lower + 1.))
        mean = numpyro.deterministic("mean", jnp.array([x]))
        numpyro.sample("obs", dist.Normal(mean, 1.), obs=jnp.array([0.75]))

    result = information_from_model_independent_normal(
        model=model, pdic={"lower": 0.25, "x": 0.75}, keys=["lower", "x"],
        mu_name="mean", observed=jnp.array([0.75]), sigma_sd=jnp.ones(1),
        param_space="unconstrained",
    )

    np.testing.assert_allclose(result["params_unconstrained"]["lower"], np.log(1./3.))
    np.testing.assert_allclose(result["params_unconstrained"]["x"], 0., atol=1e-12)
    derivative = np.array([0.25 * 0.75, 0.5 * 0.5])
    np.testing.assert_allclose(result["fisher"], np.outer(derivative, derivative))
    assert np.isfinite(result["fisher"]).all()


def test_information_from_model_independent_normal():
    jttv, _, _, pdic = read_testdata_tc()
    param_bounds = ttv_default_parameter_bounds(jttv)
    sample_keys = ["ecosw", "esinw", "pmass", "period", "tic"]

    def model_normal(sample_keys, param_bounds, uniform_ecc=False, eps=1e-4):
        import numpyro
        import numpyro.distributions as dist
        import jax.numpy as jnp

        """numpyro model for scaled parameters"""
        par = {}

        # sample parameters from priors
        for key in sample_keys:
            par[key] = numpyro.sample(key, dist.Uniform(
                param_bounds[key][0], param_bounds[key][1]))
        if "pmass" not in sample_keys:
            par["pmass"] = numpyro.deterministic(
                "pmass", jnp.exp(par["lnpmass"]))

        # Jacobian for uniform ecc prior
        # eps is introduced to prevent the log singularity at e=0; a smaller value can be used if needed
        if uniform_ecc:
            ecc = numpyro.deterministic(
                "ecc", jnp.sqrt(par['ecosw']**2+par['esinw']**2))
            numpyro.factor("eprior", -jnp.log(ecc + eps))

        # compute transit times
        tcmodel, ediff = jttv.get_transit_times_obs(par)
        numpyro.deterministic("ediff", ediff)
        numpyro.deterministic("tcmodel", tcmodel)

        # likelihood
        tcerrmodel = jttv.errorobs_flatten
        numpyro.sample("obs", dist.Normal(
            loc=tcmodel, scale=tcerrmodel), obs=jttv.tcobs_flatten)

    info_c = information_from_model_independent_normal(model=model_normal, mu_name="tcmodel", pdic=pdic, keys=sample_keys, sigma_sd=jttv.errorobs_flatten,
                                                       observed=jttv.tcobs_flatten, param_space="constrained", model_args=(sample_keys, param_bounds))['fisher']
    info_c_ref = np.loadtxt(path/"info.txt")
    assert np.allclose(info_c, info_c_ref, rtol=1e-4)

    info_u = information_from_model_independent_normal(model=model_normal, mu_name="tcmodel", pdic=pdic, keys=sample_keys, sigma_sd=jttv.errorobs_flatten,
                                                       observed=jttv.tcobs_flatten, param_space="unconstrained", model_args=(sample_keys, param_bounds))['fisher']
    info_u_ref = np.load(path/"info_unconstrained_ref.npy")
    assert np.allclose(info_u, info_u_ref, rtol=1e-4)

    sample_keys_lnpmass = ["ecosw", "esinw", "lnpmass", "period", "tic"]
    pdic['lnpmass'] = np.log(pdic['pmass'])
    info_c_lnpmass = information_from_model_independent_normal(model=model_normal, mu_name="tcmodel", pdic=pdic, keys=sample_keys_lnpmass,
                                                               sigma_sd=jttv.errorobs_flatten, observed=jttv.tcobs_flatten, param_space="constrained", model_args=(sample_keys_lnpmass, param_bounds))['fisher']
    info_c_ref_lnpmass = np.load(path/"info_lnpmass.npy")
    assert np.allclose(info_c_lnpmass, info_c_ref_lnpmass, rtol=1e-4)

    info_u_lnpmass = information_from_model_independent_normal(model=model_normal, mu_name="tcmodel", pdic=pdic, keys=sample_keys_lnpmass,
                                                               sigma_sd=jttv.errorobs_flatten, observed=jttv.tcobs_flatten, param_space="unconstrained", model_args=(sample_keys_lnpmass, param_bounds))['fisher']
    info_u_ref_lnpmass = np.load(path/"info_unconstrained_ref_lnpmass.npy")
    assert np.allclose(info_u_lnpmass, info_u_ref_lnpmass, rtol=1e-4)


if __name__ == "__main__":
    test_information_from_model_independent_normal()
