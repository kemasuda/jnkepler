from scipy.special import gammaln
import jax
__all__ = [
    "ttv_default_parameter_bounds",
    "ttv_optim_least_squares",
    "ttv_optim_curve_fit",
    "scale_pdic",
    "unscale_pdic",
    "get_flat_param_index",
    "make_phase_to_tic_transform",
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


def student_t_2nll_loss(nu, scale=1.0):
    """
    Student-t loss for scipy.optimize.least_squares (2*NLL convention).

    This callable follows SciPy's custom loss interface:
        input  : z = f**2
        output : array with shape (3, m), containing rho, rho', rho''

    Here f is the residual vector passed to least_squares. If your residuals
    are already normalized by observational errors, then `scale` is an
    additional Student-t scale parameter in those normalized units.

    The returned rho(z) is the full 2*NLL per data point, including constants.
    Therefore, with f_scale=1.0 in least_squares, res.cost corresponds to
    the total NLL.
    """
    nu = float(nu)
    scale = float(scale)

    if nu <= 0:
        raise ValueError("nu must be positive.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    a = nu * scale**2
    const = (
        2.0 * np.log(scale)
        + np.log(nu * np.pi)
        + 2.0 * gammaln(nu / 2.0)
        - 2.0 * gammaln((nu + 1.0) / 2.0)
    )

    def loss(z):
        rho = (nu + 1.0) * np.log1p(z / a) + const
        drho = (nu + 1.0) / (a + z)
        d2rho = -(nu + 1.0) / (a + z) ** 2
        return np.vstack((rho, drho, d2rho))

    return loss


def get_flat_param_index(keys, npl, key, planet_idx):
    """
    Return the flat index for a given parameter key and planet index.

    Args:
        keys: ordered list of parameter keys used in the flattened vector
        npl: total number of planets in the model
        key: parameter name, e.g. 'period', 'tic'
        planet_idx: zero-based planet index

    Returns:
        int: flat index into the concatenated parameter vector
    """
    if key not in keys:
        raise ValueError(f"{key} not found in keys={keys}")
    if not (0 <= planet_idx < npl):
        raise ValueError(f"planet_idx={planet_idx} out of range for npl={npl}")

    return keys.index(key) * npl + planet_idx


def make_phase_to_tic_transform(keys, npl, planet_idx, t_start):
    """
    Create a transform that converts the 'tic' entry for one planet from
    orbital phase to time of inferior conjunction.

    Assumes the optimizer parameter vector stores:
        tic_param = phase
    and converts it to:
        tic = phase * period + t_start

    Args:
        keys: ordered list of parameter keys in the flattened vector
        npl: total number of planets
        planet_idx: zero-based index of the target planet
        t_start: reference start time

    Returns:
        callable: transform(p_flat) -> transformed p_flat
    """
    period_idx = get_flat_param_index(keys, npl, "period", planet_idx)
    tic_idx = get_flat_param_index(keys, npl, "tic", planet_idx)

    def transform(p_flat):
        tic = p_flat[tic_idx] * p_flat[period_idx] + t_start
        return p_flat.at[tic_idx].set(tic)

    return transform


def ttv_optim_least_squares(
    jttv,
    param_bounds_,
    pinit=None,
    n_start=1,
    loss="linear",
    loss_kwargs=None,
    jac=False,
    plot=True,
    save=None,
    transit_orbit_idx=None,
    random_state=None,
    max_nfev=None,
    param_transform=None,
):
    """Simple TTV fit using scipy.optimize.least_squares.

    Args:
        jttv: JaxTTV object
        param_bounds_: bounds for parameters, dict of {key: (lower, upper)}
        pinit: initial guess of parameters (dict)
        n_start: number of initial guesses when pinit is not provided.
            If pinit is given, a single optimization is run. Otherwise,
            multi-start randomizes only the initial planet masses. The first
            start uses a deterministic midpoint-like initial guess, and later
            starts randomize only `lnpmass`.
        loss: loss specification for scipy.optimize.least_squares.
            This can be:
              - a built-in SciPy loss string such as 'linear', 'soft_l1',
                'huber', 'cauchy', 'arctan'
              - 'student_t' to use Student-t negative log likelihood
              - a custom callable compatible with scipy.optimize.least_squares
        loss_kwargs: optional dict for loss-specific options. Default is None.
            For loss='student_t':
                {'nu': ..., 'scale': ...}
                Default is {'nu': 4.0, 'scale': 1.0}.

            For built-in or custom non-linear losses:
                {'f_scale': ...} can be passed through to least_squares.

            For loss='student_t', the custom loss returns 2*NLL per data
            point, so least_squares.cost corresponds to the total NLL.
            For loss='linear', cost = 0.5 * chi2.
        jac: if True, use a JAX-based analytic Jacobian for the residual
            function. This can reduce the number of optimizer iterations and
            speed up repeated fits once the compiled function is cached, but
            the first call may take longer due to JIT compilation.
        plot: if True, TTV models are plotted with data.
        save: path to save TTV plots.
        transit_orbit_idx: list of indices to specify which planets are
            transiting (needed when non-transiting planets are included)
        random_state: int or np.random.RandomState, for reproducibility.
            If None, multi-start initializations are not reproducible across runs.
        max_nfev: maximum number of function evaluations
        param_transform: optional callable mapping optimizer-space parameters
            to model-space parameters before residual evaluation.
            This is useful, for example, when optimizing orbital phase
            instead of time of inferior conjunction for a non-transiting planet.
            If jac=True, this should be JAX-compatible.

    Returns:
        dict: best-fit JaxTTV parameter dictionary
    """
    param_bounds = deepcopy(param_bounds_)

    if loss_kwargs is None:
        loss_kwargs = {}

    # check non-transiting planets
    npl = len(param_bounds["period"][0])
    if npl != jttv.nplanet:
        print(f"# {npl - jttv.nplanet} non-transiting planets.")

        if transit_orbit_idx is None:
            raise ValueError(
                "transit_orbit_idx must be provided when non-transiting planets "
                "are included."
            )

        transit_orbit_idx = np.asarray(transit_orbit_idx)

        if transit_orbit_idx.ndim != 1:
            raise ValueError(
                f"transit_orbit_idx must be 1D, got shape {transit_orbit_idx.shape}."
            )

        if len(transit_orbit_idx) != jttv.nplanet:
            raise ValueError(
                f"transit_orbit_idx must have length {jttv.nplanet}, "
                f"got {len(transit_orbit_idx)}."
            )

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
    resid_base = cache["resid"]
    jac_resid_base = cache["jac_resid"]

    def transform_p(p):
        if param_transform is None:
            return p
        return param_transform(p)

    def resid_jax(p):
        return resid_base(transform_p(p))

    if jac:
        if param_transform is None:
            jac_resid_jax = jac_resid_base
        else:
            # include chain rule through param_transform
            jac_resid_jax = jax.jit(jax.jacfwd(resid_jax))

    def resid_np(p):
        return np.array(resid_jax(jnp.asarray(p)), dtype=float, copy=True)

    def jac_np(p):
        return np.array(jac_resid_jax(jnp.asarray(p)), dtype=float, copy=True)

    def chi2_np(p):
        r = resid_np(p)
        return float(np.sum(r**2))

    # warm up once per cache key / transform choice
    if pinit is not None:
        p_warm = np.hstack([pinit[key] for key in keys])
    else:
        p_warm = 0.5 * (params_lower + params_upper)
    p_warm = np.clip(p_warm, params_lower, params_upper)

    if (not cache["is_warmed"]) or (param_transform is not None):
        _ = resid_np(p_warm)
        if jac:
            _ = jac_np(p_warm)
        if param_transform is None:
            cache["is_warmed"] = True

    if pinit is not None and n_start != 1:
        print("# pinit is provided; ignoring n_start and running a single optimization.")
    n_start_eff = 1 if pinit is not None else n_start

    # resolve loss for least_squares
    if loss == "student_t":
        lsq_loss = student_t_2nll_loss(
            nu=loss_kwargs.get("nu", 4.0),
            scale=loss_kwargs.get("scale", 1.0),
        )
        lsq_f_scale = 1.0
    else:
        lsq_loss = loss
        lsq_f_scale = loss_kwargs.get("f_scale", 1.0)

    best_popt = None
    best_cost = np.inf
    best_chi2 = np.inf

    print(
        f"# running least squares optimization (n_start={n_start_eff}, loss={loss})..."
    )
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
                jac=jac_np if jac else "2-point",
                bounds=bounds,
                method="trf",
                loss=lsq_loss,
                f_scale=lsq_f_scale,
                max_nfev=max_nfev,
            )
        except (RuntimeError, ValueError) as e:
            print(f"# start {i}: least_squares failed ({e})")
            continue

        dt = time.time() - t0
        chi2_fin = float(np.sum(res.fun**2))
        cost_fin = float(res.cost)

        pmass0_str = np.array2string(
            np.exp(p0[mass_slice]) / 3.003e-6,
            precision=1,
            separator=", ",
        )

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

    # convert from optimizer-space to model-space parameters
    best_popt_model = np.array(
        transform_p(jnp.asarray(best_popt)),
        dtype=float,
        copy=True,
    )
    pdic_opt = params_to_dict(best_popt_model, npl, keys)

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
