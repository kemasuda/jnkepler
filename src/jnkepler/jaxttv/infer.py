
__all__ = ["ttv_default_parameter_bounds",
           "ttv_optim_curve_fit", "scale_pdic", "unscale_pdic"]

from jax import jacrev
import numpy as np
import jax.numpy as jnp
from scipy.optimize import curve_fit
from copy import deepcopy
import time
import warnings
from .utils import params_to_dict


def ttv_default_parameter_bounds(jttv, dtic=0.05, dp_frac=1e-2, emax=0.2, mmin=1e-7, mmax=1e-3):
    """get parameter bounds for TTV optimization

        Args:
            jttv: JaxTTV object

        Returns:
            dict: dictionary of (lower bound array, upper bound array)

    """
    npl = jttv.nplanet
    t0_guess = np.array([tcobs_[0] for tcobs_ in jttv.tcobs])
    p_guess = np.array(jttv.p_init)

    ones = np.ones(npl)
    evector_shift = 1e-2  # avoid (0,0) for initial (ecosw, esinw)
    param_bounds = {
        "tic": [t0_guess - dtic, t0_guess + dtic],
        "period": [p_guess * (1 - dp_frac), p_guess * (1 + dp_frac)],
        "ecosw": [-emax * ones, emax * ones - evector_shift],
        "esinw": [-emax * ones, emax * ones - evector_shift],
        "lnpmass": [np.log(mmin) * ones, np.log(mmax) * ones],
        "pmass": [mmin * ones, mmax * ones],
    }

    return param_bounds


def scale_pdic(pdic, param_bounds):
    """scale parameters using bounds

        Args:
            pdic: dict of physical parameters
            param_bounds: dictionary of (lower bound array, upper bound array)

        Returns:
            dict: dictionary of scaled parameters

    """
    pdic_scaled = {}
    for key in param_bounds.keys():
        pdic_scaled[key+"_scaled"] = (pdic[key] - param_bounds[key][0]) / \
            (param_bounds[key][1] - param_bounds[key][0])
    return pdic_scaled


def unscale_pdic(pdic_scaled, param_bounds):
    """unscale parameters using bounds

        Args:
            pdic: dict of scaled parameters
            param_bounds: dictionary of (lower bound array, upper bound array)

        Returns:
            dict: dictionary of physical parameters in original scales

    """
    pdic = {}
    for key in param_bounds.keys():
        pdic[key] = param_bounds[key][0] + \
            (param_bounds[key][1] - param_bounds[key][0]) * \
            pdic_scaled[key+"_scaled"]
    return pdic


def ttv_optim_curve_fit(jttv, param_bounds_, p_init=None, jac=False, plot=True, save=None):
    """simple TTV fit using scipy.curve_fit with bounds

        Args:
            jttv: JaxTTV object
            param_bounds: bounds for parameters, 0: lower, 1: upper
            p_init: initial parameter values (if None, center of lower/upper bounds)
            jac: if True jacrev(model) is used

        Returns:
            dict: JaxTTV parameter dictionary

    """
    npl = jttv.nplanet
    param_bounds = deepcopy(param_bounds_)

    if "cosi" not in param_bounds.keys() or "lnode" not in param_bounds.keys():
        warnings.warn(
            "Bounds for cosi/lnode not provided: assuming coplanar orbits.")
        keys = ['period', 'ecosw', 'esinw', 'tic', "lnpmass"]
    else:
        keys = ['period', 'ecosw', 'esinw', 'cosi', 'lnode', 'tic', "lnpmass"]
    params_lower = np.hstack([param_bounds[key][0] for key in keys])
    params_upper = np.hstack([param_bounds[key][1] for key in keys])

    bounds = (params_lower, params_upper)

    if p_init is None:
        p_init = 0.5 * (params_lower + params_upper)

    def model(p): return jttv.get_transit_times_obs(
        params_to_dict(p, npl, keys))[0]
    func = lambda x, *params: model(jnp.array(params))

    def objective(p): return jnp.sum(
        (model(p) - jttv.tcobs_flatten)**2 / jttv.errorobs_flatten**2)
    if jac:
        jacmodel = jacrev(model)
        jacfunc = lambda x, *params: jacmodel(jnp.array(params))
    else:
        jacfunc = None

    print("# running least squares optimization...")
    start_time = time.time()
    popt, pcov = curve_fit(func, None, jttv.tcobs_flatten, p0=p_init,
                           sigma=jttv.errorobs_flatten, bounds=bounds, jac=jacfunc)
    print("# objective function: %.2f --> %.2f (%d data)" %
          (objective(p_init), objective(popt), len(jttv.tcobs_flatten)))
    print("# elapsed time: %.1f sec" % (time.time()-start_time))

    pdic_opt = params_to_dict(popt, npl, keys)

    if plot:
        tcall = jttv.get_transit_times_all_list(pdic_opt)
        jttv.plot_model(tcall, marker='.', save=save)

    pdic_opt['pmass'] = jnp.exp(pdic_opt['lnpmass'])

    return pdic_opt
