
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


def ttv_optim_curve_fit(
    jttv,
    param_bounds_,
    pinit=None,
    n_start=1,
    loss='linear',
    jac=False,
    plot=True,
    save=None,
    transit_orbit_idx=None,
    random_state=None,
    max_nfev=None,
):
    """simple TTV fit using scipy.curve_fit with multiple random starts.

    Args:
        jttv: JaxTTV object
        param_bounds_: bounds for parameters, dict of {key: (lower, upper)}
        pinit: initial guess of parameters (dict)
        n_start: number of random initial guesses
        loss: determins the loss in scipy.optimize.least_squares. 
            Using robust loss functions (e.g., 'soft_l1', 'huber') someimtes helps to mitigate the impact of outliers.
        jac: if True, use jacrev(model) as in single-start version
        plot: if True, TTV models are plotted with data.
        save: path to save TTV plots.
        transit_orbit_idx: list of indices to specify which planets are transiting (needed when non-transiting planets are included)
        random_state: int or np.random.RandomState, for reproducibility

    Returns:
        dict: best-fit JaxTTV parameter dictionary (over all starts)
    """

    param_bounds = deepcopy(param_bounds_)

    # check non-transiting planets
    npl = len(param_bounds["period"][0])
    if npl != jttv.nplanet:
        print(f"# {npl - jttv.nplanet} non-transiting planets.")
        assert len(transit_orbit_idx) == jttv.nplanet

    # keys to optimize
    if "cosi" not in param_bounds.keys() or "lnode" not in param_bounds.keys():
        warnings.warn(
            "Bounds for cosi/lnode not provided: assuming coplanar orbits."
        )
        keys = ["period", "ecosw", "esinw", "tic", "lnpmass"]
    else:
        keys = ["period", "ecosw", "esinw", "cosi", "lnode", "tic", "lnpmass"]

    params_lower = np.hstack([param_bounds[key][0] for key in keys])
    params_upper = np.hstack([param_bounds[key][1] for key in keys])
    bounds = (params_lower, params_upper)
    ndim = params_lower.size

    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    def model(p_flat):
        pdic = params_to_dict(p_flat, npl, keys)
        return jttv.get_transit_times_obs(
            pdic, transit_orbit_idx=transit_orbit_idx
        )[0]

    func = lambda x, *params: model(jnp.array(params))

    def objective(p_flat):
        resid = (model(p_flat) - jttv.tcobs_flatten) / jttv.errorobs_flatten
        return float(jnp.sum(resid**2))

    if jac:
        jacmodel = jacrev(model)
        jacfunc = lambda x, *params: jacmodel(jnp.array(params))
    else:
        jacfunc = None

    best_popt = None
    best_obj = np.inf
    best_pcov = None

    print(
        f"# running least squares optimization (n_start={n_start})...")
    t0_all = time.time()

    for i in range(n_start):
        if pinit is not None:
            p0 = np.hstack([pinit[key] for key in keys])
        else:
            if i == 0:
                p0 = 0.499 * params_lower + 0.501 * params_upper
            else:
                # uniform
                u = rng.rand(ndim)
                p0 = params_lower + u * (params_upper - params_lower)
                '''
                mid = 0.5 * (params_lower + params_upper)
                width = params_upper - params_lower
                u = rng.normal(loc=mid, scale=0.1 * width)
                p0 = np.clip(u, params_lower, params_upper)
                '''

        t0 = time.time()
        try:
            popt, pcov = curve_fit(
                func,
                None,
                jttv.tcobs_flatten,
                p0=p0,
                sigma=jttv.errorobs_flatten,
                bounds=bounds,
                jac=jacfunc,
                max_nfev=max_nfev,
                loss=loss,
            )
        except (RuntimeError, ValueError) as e:
            print(f"#   start {i}: curve_fit failed ({e})")
            continue

        obj = objective(popt)
        dt = time.time() - t0
        print(
            f"#   start {i}: objective={objective(p0):.2f} --> {obj:.2f}, elapsed={dt:.1f} s")

        if obj < best_obj:
            best_obj = obj
            best_popt = popt
            best_pcov = pcov

    print("# ------------------------------------------------------------")
    print(
        "# best objective over all starts: %.2f (%d data)"
        % (best_obj, len(jttv.tcobs_flatten))
    )
    print("# total elapsed time: %.1f sec" % (time.time() - t0_all))
    print("# ------------------------------------------------------------")

    if best_popt is None:
        raise RuntimeError("All multi-start fits failed.")

    pdic_opt = params_to_dict(best_popt, npl, keys)

    if plot:
        tcall = jttv.get_transit_times_all_list(
            pdic_opt, transit_orbit_idx=transit_orbit_idx
        )
        jttv.plot_model(tcall, marker=".", save=save)

    pdic_opt["pmass"] = jnp.exp(pdic_opt["lnpmass"])

    return pdic_opt
