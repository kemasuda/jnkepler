from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from jnkepler.jaxttv import infer


def custom_loss(z):
    return np.vstack([z, np.ones_like(z), np.zeros_like(z)])


@pytest.mark.parametrize("loss", ["soft_l1", "huber", "cauchy", "arctan", "student_t", custom_loss])
def test_gaussian_priors_reject_non_linear_losses_before_evaluation(loss):
    with pytest.raises(ValueError, match="gaussian_priors requires loss='linear'"):
        infer.ttv_optim_least_squares(
            None, {}, loss=loss, gaussian_priors={"period": (0, 1., 1.)})


@pytest.mark.parametrize("loss, priors, expected", [
    ("linear", {"period": (0, 1., 1.)}, 7.),
    ("soft_l1", None, 10.),
])
def test_allowed_losses_optimize_the_intended_objective(monkeypatch, loss, priors, expected):
    model = SimpleNamespace(nplanet=1, transit_time_method="fast",
                            tcobs_flatten=np.array([10., 10.]),
                            errorobs_flatten=np.ones(2))

    def linear_times(jttv, par_dict, **kwargs):
        return jnp.repeat(par_dict["period"], 2), 0.

    monkeypatch.setattr(infer, "_get_transit_times_obs_with_method", linear_times)
    bounds = {"period": [np.array([.1]), np.array([20.])],
              "ecosw": [np.array([-.1]), np.array([.1])],
              "esinw": [np.array([-.1]), np.array([.1])],
              "cosi": [np.array([-.1]), np.array([.1])],
              "lnode": [np.array([-.1]), np.array([.1])],
              "tic": [np.array([0.]), np.array([1.])],
              "lnpmass": [np.array([-20.]), np.array([-10.])]}
    result = infer.ttv_optim_least_squares(
        model, bounds, loss=loss, gaussian_priors=priors, jac=True, plot=False)
    # Two unit-error observations at 10 and one N(1,1) prior have mean 7.
    np.testing.assert_allclose(result["period"], expected, atol=1e-6, rtol=0.)
