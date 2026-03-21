__all__ = [
    "ttv_default_parameter_bounds",
    "ttv_optim_least_squares",
    "ttv_optim_curve_fit",
    "scale_pdic",
    "unscale_pdic",
]

from jax import jacfwd, jit
import numpy as np
import jax.numpy as jnp
from scipy.optimize import least_squares
from copy import deepcopy
import time
import warnings

from .utils import params_to_dict, dict_to_params


def ttv_default_parameter_bounds(jttv, npl=None, t0_guess=None, p_guess=None,
                                 dtic=0.2, dp_frac=1e-2, emax=0.2,
                                 mmin=1e-7, mmax=1e-3):
    """Get parameter bounds for TTV optimization.

    Args:
        jttv: JaxTTV object.
        npl (int, optional): Number of planets. Defaults to jttv.nplanet if None.
        t0_guess (array-like, optional): Initial guess for transit times,
            length must be npl.
        p_guess (array-like, optional): Initial guess for orbital periods,
            length must be npl.
        dtic (float, optional): Half-width of bounds around t0_guess for
            transit time.
        dp_frac (float, optional): Fractional width of bounds around p_guess
            for period.
        emax (float, optional): Maximum ecosw/esinw bound.
        mmin (float, optional): Minimum mass bound.
        mmax (float, optional): Maximum mass bound.

    Returns:
        dict: Dictionary of parameter bounds with keys as parameter names
            and values as [lower_bound_array, upper_bound_array].
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
    assert len(p_guess) == npl, f"p_guess length {len(p_guess)} != npl {npl}"

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
        pdic_scaled[key + "_scaled"] = (
            (pdic[key] - param_bounds[key][0])
            / (param_bounds[key][1] - param_bounds[key][0])
        )
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
        pdic[key] = (
            param_bounds[key][0]
            + (param_bounds[key][1] - param_bounds[key][0]) *
            pdic_scaled[key + "_scaled"]
        )
    return pdic


def _get_cached_residual_functions(jttv, npl, keys, transit_orbit_idx=None, jac=False):
    """Return cached jitted residual / jacobian functions for repeated fits."""
    if not hasattr(jttv, "_lsq_cache"):
        jttv._lsq_cache = {}

    cache_key = (
        npl,
        tuple(keys),
        None if transit_orbit_idx is None else tuple(transit_orbit_idx),
        bool(jac),
    )

    if cache_key not in jttv._lsq_cache:
        def _model(p_flat):
            pdic = params_to_dict(p_flat, npl, keys)
            return jttv.get_transit_times_obs(
                pdic,
                transit_orbit_idx=transit_orbit_idx,
            )[0]

        def _resid(p_flat):
            return (_model(p_flat) - jttv.tcobs_flatten) / jttv.errorobs_flatten

        resid = jit(_resid)
        jac_resid = jit(jacfwd(_resid)) if jac else None

        jttv._lsq_cache[cache_key] = {
            "resid": resid,
            "jac_resid": jac_resid,
            "is_warmed": False,
        }

    return jttv._lsq_cache[cache_key]


def ttv_optim_least_squares(
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
    """Simple TTV fit using scipy.optimize.least_squares.

    Args:
        jttv: JaxTTV object
        param_bounds_: bounds for parameters, dict of {key: (lower, upper)}
        pinit: initial guess of parameters (dict)
        n_start: number of initial guesses when pinit is not provided.
            If pinit is given, a single optimization is run. Otherwise,
            multi-start randomizes only the initial planet masses.
        loss: determines the loss in scipy.optimize.least_squares.
            Using robust loss functions (e.g., 'soft_l1', 'huber') sometimes
            helps to mitigate the impact of outliers.
        jac: if True, use a JAX-based analytic Jacobian for the residual function.
            This can reduce the number of optimizer iterations and speed up
            repeated fits once the compiled function is cached, but the first
            call may take longer due to JIT compilation. The default is False
            to avoid this initial overhead in simple one-off fits.
        plot: if True, TTV models are plotted with data.
        save: path to save TTV plots.
        transit_orbit_idx: list of indices to specify which planets are transiting
            (needed when non-transiting planets are included)
        random_state: int or np.random.RandomState, for reproducibility
        max_nfev: maximum number of function evaluations

    Returns:
        dict: best-fit JaxTTV parameter dictionary
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

    # slice for lnpmass in the flattened parameter vector
    offset = 0
    mass_slice = None
    for key in keys:
        n = len(param_bounds[key][0])
        if key == "lnpmass":
            mass_slice = slice(offset, offset + n)
            break
        offset += n

    if mass_slice is None:
        raise ValueError("lnpmass not found in optimization keys.")

    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    cache = _get_cached_residual_functions(
        jttv,
        npl=npl,
        keys=keys,
        transit_orbit_idx=transit_orbit_idx,
        jac=jac,
    )
    resid = cache["resid"]
    jac_resid = cache["jac_resid"]

    def resid_np(p):
        return np.asarray(resid(jnp.asarray(p)))

    def jac_np(p):
        return np.asarray(jac_resid(jnp.asarray(p)))

    def chi2_np(p):
        r = resid_np(p)
        return float(np.sum(r**2))

    # warm up once per cache key
    if pinit is not None:
        p_warm = np.hstack([pinit[key] for key in keys])
    else:
        p_warm = 0.5 * (params_lower + params_upper)
    p_warm = np.clip(p_warm, params_lower, params_upper)

    if not cache["is_warmed"]:
        _ = resid_np(p_warm)
        if jac:
            _ = jac_np(p_warm)
        cache["is_warmed"] = True

    if pinit is not None and n_start != 1:
        print("# pinit is provided; ignoring n_start and running a single optimization.")
    n_start_eff = 1 if pinit is not None else n_start

    best_popt = None
    best_cost = np.inf
    best_chi2 = np.inf

    print(f"# running least squares optimization (n_start={n_start_eff})...")
    t0_all = time.time()

    # baseline starting point used when pinit is not given
    p0_base = 0.499 * params_lower + 0.501 * params_upper

    for i in range(n_start_eff):
        if pinit is not None:
            p0 = np.hstack([pinit[key] for key in keys])
        else:
            p0 = p0_base.copy()
            if i > 0:
                lo = params_lower[mass_slice]
                hi = params_upper[mass_slice]
                p0[mass_slice] = lo + rng.rand(hi.size) * (hi - lo)

        p0 = np.clip(p0, params_lower, params_upper)

        chi2_init = chi2_np(p0)
        t0 = time.time()

        try:
            res = least_squares(
                resid_np,
                p0,
                jac=jac_np if jac else '2-point',
                bounds=bounds,
                method='trf',
                loss=loss,
                max_nfev=max_nfev,
            )
        except (RuntimeError, ValueError) as e:
            print(f"# start {i}: least_squares failed ({e})")
            continue

        dt = time.time() - t0
        chi2_fin = float(np.sum(res.fun**2))
        cost_fin = float(res.cost)

        pmass0_str = np.array2string(
            np.exp(p0[mass_slice])/3.003e-6, precision=1, separator=", ")

        print(
            f"# start {i}: initial pmass={pmass0_str}, "
            f"chi2={chi2_init:.2f} --> {chi2_fin:.2f}, "
            f"cost={cost_fin:.2f}, nfev={res.nfev}, elapsed={dt:.1f} s"
        )

        if cost_fin < best_cost:
            best_cost = cost_fin
            best_chi2 = chi2_fin
            best_popt = res.x

    print("# ------------------------------------------------------------")
    print(
        "# best objective over all starts: "
        f"cost={best_cost:.2f}, chi2={best_chi2:.2f} "
        f"({len(jttv.tcobs_flatten)} data)"
    )
    print("# total elapsed time: %.1f sec" % (time.time() - t0_all))
    print("# ------------------------------------------------------------")

    if best_popt is None:
        raise RuntimeError("All fits failed.")

    pdic_opt = params_to_dict(best_popt, npl, keys)

    if plot:
        tcall = jttv.get_transit_times_all_list(
            pdic_opt,
            transit_orbit_idx=transit_orbit_idx,
        )
        jttv.plot_model(tcall, marker=".", save=save)

    pdic_opt["pmass"] = jnp.exp(pdic_opt["lnpmass"])
    return pdic_opt


def ttv_optim_curve_fit(*args, **kwargs):
    """Deprecated wrapper for ttv_optim_least_squares."""
    warnings.warn(
        "ttv_optim_curve_fit is deprecated and will be removed in a future "
        "release. Use ttv_optim_least_squares instead.",
        FutureWarning,
        stacklevel=2,
    )
    return ttv_optim_least_squares(*args, **kwargs)
