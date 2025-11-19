
__all__ = ["ttv_default_parameter_bounds",
           "ttv_optim_curve_fit", "scale_pdic", "unscale_pdic"]

from jax import jacrev
import numpy as np
import jax.numpy as jnp
from scipy.optimize import curve_fit
from copy import deepcopy
import time
import warnings
from .utils import params_to_dict, dict_to_params


def ttv_default_parameter_bounds(jttv, npl=None, t0_guess=None, p_guess=None, dtic=0.2, dp_frac=1e-2, emax=0.2, mmin=1e-7, mmax=1e-3):
    """Get parameter bounds for TTV optimization.

    Args:
        jttv: JaxTTV object.
        npl (int, optional): Number of planets. Defaults to jttv.nplanet if None.
        t0_guess (array-like, optional): Initial guess for transit times, length must be npl.
        p_guess (array-like, optional): Initial guess for orbital periods, length must be npl.
        dtic (float, optional): Half-width of bounds around t0_guess for transit time.
        dp_frac (float, optional): Fractional width of bounds around p_guess for period.
        emax (float, optional): Maximum ecosw/esinw bound.
        mmin (float, optional): Minimum mass bound.
        mmax (float, optional): Maximum mass bound.

    Returns:
        dict: Dictionary of parameter bounds with keys as parameter names and values as [lower_bound_array, upper_bound_array].

    """
    if npl is None:
        npl = jttv.nplanet

    if t0_guess is None:
        t0_guess = np.array([tcobs_[0] for tcobs_ in jttv.tcobs])
    else:
        t0_guess = np.array(t0_guess)
        assert len(
            t0_guess) == npl, f"t0_guess length {len(t0_guess)} != npl {npl}"

    if p_guess is None:
        p_guess = np.array(jttv.p_init)
    else:
        p_guess = np.array(p_guess)
        assert len(
            p_guess) == npl, f"p_guess length {len(p_guess)} != npl {npl}"

    ones = np.ones(npl)
    param_bounds = {
        "tic": [t0_guess - dtic, t0_guess + dtic],
        "period": [p_guess * (1 - dp_frac), p_guess * (1 + dp_frac)],
        "ecosw": [-emax * ones, emax * ones],
        "esinw": [-emax * ones, emax * ones],
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


def ttv_optim_curve_fit(jttv, param_bounds_, p_init=None, jac=False, plot=True, save=None, transit_orbit_idx=None):
    """simple TTV fit using scipy.curve_fit with bounds.

        Args:
            jttv: JaxTTV object
            param_bounds: bounds for parameters, 0: lower, 1: upper
            p_init: dictionary containing initial parameter values (if None, center of lower/upper bounds)
            jac: if True jacrev(model) is used
            transit_orbit_idx: list of indices to specify which planets are transiting (needed when non-transiting planets are included)

        Returns:
            dict: JaxTTV parameter dictionary

        Note:
            TTV fitting may admit multiple local minima, but this function does not attempt to identify all possible solutions.

    """
    npl = len(param_bounds_['period'][0])
    if npl != jttv.nplanet:
        print(f"# {npl-jttv.nplanet} non-transiting planets.")
        assert len(transit_orbit_idx) == jttv.nplanet
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
        p_init = 0.499 * params_lower + 0.501 * params_upper
    else:
        p_init = dict_to_params(p_init, npl, keys)

    def model(p): return jttv.get_transit_times_obs(
        params_to_dict(p, npl, keys), transit_orbit_idx=transit_orbit_idx)[0]
    func = lambda x, *params: model(jnp.array(params))

    def objective(p):
        return jnp.sum((model(p) - jttv.tcobs_flatten)**2 / jttv.errorobs_flatten**2)
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
        tcall = jttv.get_transit_times_all_list(
            pdic_opt, transit_orbit_idx=transit_orbit_idx)
        jttv.plot_model(tcall, marker='.', save=save)

    pdic_opt['pmass'] = jnp.exp(pdic_opt['lnpmass'])

    return pdic_opt
