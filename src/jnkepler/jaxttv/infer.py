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

from jax import jacfwd, jacrev, jit
import numpy as np
import jax.numpy as jnp
from scipy.optimize import least_squares
from copy import deepcopy
import time
import warnings

from .symplectic import integrate_xv
from .utils import (
    dict_to_params,
    get_energy_diff_jac,
    initialize_jacobi_xv,
    params_to_dict,
)


def _canonicalize_transit_time_method(method):
    """Return the canonical transit-time method name."""
    if method is None:
        return None
    if method == "newton-raphson":
        warnings.warn(
            "'newton-raphson' is deprecated; use 'newton' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "newton"
    elif method == "interpolation":
        warnings.warn(
            "'interpolation' is deprecated; use 'kepler' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "kepler"

    if method not in ("fast", "newton", "kepler"):
        raise ValueError(
            "transit_time_method must be one of 'fast', 'newton', or 'kepler'."
        )
    return method


def _get_transit_times_obs_with_method(
    jttv,
    par_dict,
    transit_orbit_idx=None,
    transit_time_method=None,
):
    """Compute observed transit times with an explicit transit-time method."""
    if transit_time_method is None:
        transit_time_method = jttv.transit_time_method

    xjac0, vjac0, masses = initialize_jacobi_xv(par_dict, jttv.t_start)
    times, xvjac = integrate_xv(
        xjac0, vjac0, masses, jttv.times, nitr=jttv.nitr_kepler)

    if transit_orbit_idx is None:
        orbit_idx = jttv.pidx.astype(int) - 1
    else:
        orbit_idx = transit_orbit_idx[jttv.pidx.astype(int) - 1].astype(int)

    transit_times = jttv._compute_transit_times(
        orbit_idx,
        jttv.tcobs_flatten,
        times,
        xvjac,
        masses,
        method=transit_time_method,
    )
    ediff = get_energy_diff_jac(xvjac, masses, -0.5 * jttv.dt)
    return transit_times, ediff


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


def _get_cached_residual_functions(
    jttv,
    npl,
    keys,
    transit_orbit_idx=None,
    transit_time_method=None,
    jac=False,
    diff_mode="fwd",
):
    """Return cached jitted residual / jacobian functions for repeated fits."""
    if not hasattr(jttv, "_lsq_cache"):
        jttv._lsq_cache = {}

    if diff_mode not in ("fwd", "rev"):
        raise ValueError(
            f"diff_mode must be 'fwd' or 'rev', got {diff_mode!r}"
        )

    cache_key = (
        npl,
        tuple(keys),
        None if transit_orbit_idx is None else tuple(transit_orbit_idx),
        transit_time_method,
        bool(jac),
        diff_mode,
    )

    if cache_key not in jttv._lsq_cache:
        def _model(p_flat):
            pdic = params_to_dict(p_flat, npl, keys)
            return _get_transit_times_obs_with_method(
                jttv,
                pdic,
                transit_orbit_idx=transit_orbit_idx,
                transit_time_method=transit_time_method,
            )[0]

        def _resid(p_flat):
            return (_model(p_flat) - jttv.tcobs_flatten) / jttv.errorobs_flatten

        resid = jit(_resid)

        if jac:
            if diff_mode == "fwd":
                jac_resid = jit(jacfwd(_resid))
            else:
                jac_resid = jit(jacrev(_resid))
        else:
            jac_resid = None

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
    diff_mode="fwd",
    plot=True,
    save=None,
    transit_orbit_idx=None,
    random_state=None,
    max_nfev=None,
    param_transform=None,
    return_optspace=False,
    gaussian_priors=None,
    transit_time_method="newton",
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

            Gaussian priors are currently not supported with loss='student_t'.
            With other non-linear losses, the Gaussian-prior residuals are
            passed to scipy.optimize.least_squares in the same residual vector
            as the data residuals.
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
            function.
        diff_mode: differentiation mode for the analytic Jacobian when
            jac=True. Defaults to 'fwd' (``jax.jacfwd``). Must be one of:
              - 'fwd': use ``jax.jacfwd``
              - 'rev': use ``jax.jacrev``
              - 'auto': use 'fwd' for transit_time_method='fast' and
                'rev' for transit_time_method='newton'
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
        return_optspace: if True, also return the best-fit parameter dictionary
            in optimizer space as a second output. This is useful for warm
            starts when param_transform is used.
        gaussian_priors: optional Gaussian priors on optimizer-space parameters.
            This is a dict of {key: prior_spec}. Each prior_spec can be either

                {'index': idx, 'mean': mean, 'sigma': sigma}

            or

                (idx, mean, sigma)

            where idx can be an int or a list/array of indices. If idx is None
            or omitted, the prior is applied to all elements of that parameter.

            The important point is that the prior is applied before
            param_transform. Therefore, if param_transform maps optimizer-space
            parameters to model-space parameters, gaussian_priors refers to the
            optimizer-space values, not the transformed model-space values.

            The added residual is

                (p_opt - mean) / sigma.

            Example:

                gaussian_priors={
                    'period': {'index': 7, 'mean': 60.0, 'sigma': 1.0},
                    'lnpmass': {'index': 7, 'mean': np.log(3e-5), 'sigma': 0.5},
                }
        transit_time_method: algorithm used for transit-time computation during
            least-squares optimization. Defaults to 'newton'. Supported values
            are 'fast', 'newton', and 'kepler'. Set to None to use
            ``jttv.transit_time_method``.

    Returns:
        dict or tuple:
            - If return_optspace=False (default), returns the best-fit JaxTTV
              parameter dictionary in model space.
            - If return_optspace=True, returns
              (pdic_opt_modelspace, pdic_opt_optspace).
    """
    param_bounds = deepcopy(param_bounds_)

    if loss_kwargs is None:
        loss_kwargs = {}

    if gaussian_priors is not None and loss == "student_t":
        raise NotImplementedError(
            "gaussian_priors is currently not supported with loss='student_t'."
        )

    if diff_mode not in ("auto", "rev", "fwd"):
        raise ValueError(
            f"diff_mode must be 'auto', 'rev', or 'fwd', got {diff_mode!r}"
        )

    # resolve transit-time and differentiation modes
    transit_time_method = _canonicalize_transit_time_method(transit_time_method)
    if transit_time_method is None:
        transit_time_method = jttv.transit_time_method

    if diff_mode == "auto":
        effective_diff_mode = (
            "rev" if transit_time_method == "newton" else "fwd"
        )
    else:
        effective_diff_mode = diff_mode

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

    # slices in the flattened optimizer-space parameter vector
    offset = 0
    key_slices = {}
    mass_slice = None

    for key in keys:
        n = len(param_bounds[key][0])
        key_slices[key] = slice(offset, offset + n)

        if key == "lnpmass":
            mass_slice = key_slices[key]

        offset += n

    if mass_slice is None:
        raise ValueError("lnpmass not found in optimization keys.")

    def _parse_gaussian_priors(gaussian_priors):
        """Parse Gaussian priors on optimizer-space parameters."""
        if gaussian_priors is None:
            return []

        prior_terms = []

        for key, spec in gaussian_priors.items():
            if key not in key_slices:
                raise ValueError(
                    f"Gaussian prior key {key!r} is not in optimized keys: {keys}."
                )

            key_slice = key_slices[key]
            n_key = key_slice.stop - key_slice.start

            if isinstance(spec, dict):
                idx = spec.get("index", spec.get("idx", spec.get("indices", None)))
                mean = spec["mean"]
                sigma = spec["sigma"]
            else:
                if len(spec) != 3:
                    raise ValueError(
                        "Gaussian prior spec must be either a dict with "
                        "'index', 'mean', and 'sigma', or a tuple "
                        "(index, mean, sigma)."
                    )
                idx, mean, sigma = spec

            if idx is None:
                idx = np.arange(n_key, dtype=int)
            elif np.isscalar(idx):
                idx = np.array([idx], dtype=int)
            else:
                idx = np.asarray(idx, dtype=int)

            if np.any(idx < 0) or np.any(idx >= n_key):
                raise ValueError(
                    f"Gaussian prior index out of range for {key!r}: {idx}."
                )

            mean = np.asarray(mean, dtype=float)
            sigma = np.asarray(sigma, dtype=float)

            if mean.ndim == 0:
                mean = np.full(idx.size, float(mean))
            if sigma.ndim == 0:
                sigma = np.full(idx.size, float(sigma))

            if mean.shape != (idx.size,):
                raise ValueError(
                    f"mean for Gaussian prior {key!r} must have shape "
                    f"{(idx.size,)}, got {mean.shape}."
                )

            if sigma.shape != (idx.size,):
                raise ValueError(
                    f"sigma for Gaussian prior {key!r} must have shape "
                    f"{(idx.size,)}, got {sigma.shape}."
                )

            if np.any(sigma <= 0):
                raise ValueError(f"sigma must be positive for Gaussian prior {key!r}.")

            flat_idx = key_slice.start + idx

            prior_terms.append(
                (
                    jnp.asarray(flat_idx, dtype=int),
                    jnp.asarray(mean),
                    jnp.asarray(sigma),
                )
            )

        return prior_terms

    gaussian_prior_terms = _parse_gaussian_priors(gaussian_priors)
    n_gaussian_prior = sum(len(idx) for idx, _, _ in gaussian_prior_terms)

    if isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    cache = _get_cached_residual_functions(
        jttv,
        npl=npl,
        keys=keys,
        transit_orbit_idx=transit_orbit_idx,
        transit_time_method=transit_time_method,
        jac=jac,
        diff_mode=effective_diff_mode,
    )
    resid_base = cache["resid"]
    jac_resid_base = cache["jac_resid"]

    def transform_p(p):
        if param_transform is None:
            return p
        return param_transform(p)

    def gaussian_prior_resid_jax(p_opt):
        """Gaussian-prior residuals in optimizer space."""
        if len(gaussian_prior_terms) == 0:
            return jnp.zeros((0,), dtype=p_opt.dtype)

        return jnp.concatenate(
            [
                (p_opt[idx] - mean) / sigma
                for idx, mean, sigma in gaussian_prior_terms
            ]
        )

    def resid_data_jax(p):
        """Data residuals only."""
        return resid_base(transform_p(p))

    def resid_jax(p):
        """Data residuals plus Gaussian-prior residuals.

        The data model sees transform_p(p), but Gaussian priors are applied
        directly to p, i.e. before param_transform.
        """
        r_data = resid_data_jax(p)
        r_prior = gaussian_prior_resid_jax(p)
        return jnp.concatenate([r_data, r_prior])

    if jac:
        if (
            param_transform is None
            and len(gaussian_prior_terms) == 0
        ):
            jac_resid_jax = jac_resid_base
        else:
            # include chain rule through param_transform and prior residuals
            if effective_diff_mode == "rev":
                jac_resid_jax = jax.jit(jax.jacrev(resid_jax))
            else:
                jac_resid_jax = jax.jit(jax.jacfwd(resid_jax))

    def resid_np(p):
        return np.array(resid_jax(jnp.asarray(p)), dtype=float, copy=True)

    def resid_data_np(p):
        return np.array(resid_data_jax(jnp.asarray(p)), dtype=float, copy=True)

    def resid_prior_np(p):
        return np.array(
            gaussian_prior_resid_jax(jnp.asarray(p)),
            dtype=float,
            copy=True,
        )

    def jac_np(p):
        return np.array(jac_resid_jax(jnp.asarray(p)), dtype=float, copy=True)

    def chi2_data_np(p):
        r = resid_data_np(p)
        return float(np.sum(r**2))

    def chi2_prior_np(p):
        r = resid_prior_np(p)
        return float(np.sum(r**2))

    # warm up once per cache key / transform choice
    if pinit is not None:
        p_warm = np.hstack([pinit[key] for key in keys])
    else:
        p_warm = 0.5 * (params_lower + params_upper)
    p_warm = np.clip(p_warm, params_lower, params_upper)

    if (
        (not cache["is_warmed"])
        or (param_transform is not None)
        or (len(gaussian_prior_terms) > 0)
    ):
        _ = resid_np(p_warm)
        if jac:
            _ = jac_np(p_warm)
        if param_transform is None and len(gaussian_prior_terms) == 0:
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
    best_chi2_data = np.inf
    best_chi2_prior = np.inf

    print(
        "# running least squares optimization "
        f"(n_start={n_start_eff}, loss={loss}, "
        f"n_gaussian_prior={n_gaussian_prior}, "
        f"transit_time_method={transit_time_method}, "
        f"jac={'on' if jac else 'off'}, "
        f"diff_mode={effective_diff_mode if jac else 'n/a'})..."
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

        chi2_data_init = chi2_data_np(p0)
        chi2_prior_init = chi2_prior_np(p0)
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
        chi2_data_fin = chi2_data_np(res.x)
        chi2_prior_fin = chi2_prior_np(res.x)
        cost_fin = float(res.cost)

        pmass0_str = np.array2string(
            np.exp(p0[mass_slice]) / 3.003e-6,
            precision=1,
            separator=", ",
        )

        print(
            f"# start {i}: initial pmass={pmass0_str}, "
            f"data_chi2={chi2_data_init:.2f} --> {chi2_data_fin:.2f}, "
            f"prior_chi2={chi2_prior_init:.2f} --> {chi2_prior_fin:.2f}, "
            f"cost={cost_fin:.2f}, nfev={res.nfev}, elapsed={dt:.1f} s"
        )

        if cost_fin < best_cost:
            best_cost = cost_fin
            best_chi2_data = chi2_data_fin
            best_chi2_prior = chi2_prior_fin
            best_popt = res.x

    print("# ------------------------------------------------------------")
    print(
        "# best objective over all starts: "
        f"cost={best_cost:.2f}, "
        f"data_chi2={best_chi2_data:.2f}, "
        f"prior_chi2={best_chi2_prior:.2f} "
        f"({len(jttv.tcobs_flatten)} data + {n_gaussian_prior} Gaussian priors)"
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

    if transit_time_method == "fast":
        fast_validation_threshold = float(
            np.median(np.asarray(jttv.errorobs_flatten, dtype=float)))

        tc_fast = np.asarray(
            _get_transit_times_obs_with_method(
                jttv,
                pdic_opt,
                transit_orbit_idx=transit_orbit_idx,
                transit_time_method="fast",
            )[0],
            dtype=float,
        )

        tc_newton = np.asarray(
            _get_transit_times_obs_with_method(
                jttv,
                pdic_opt,
                transit_orbit_idx=transit_orbit_idx,
                transit_time_method="newton",
            )[0],
            dtype=float,
        )

        max_abs_dt = float(np.max(np.abs(tc_fast - tc_newton)))

        if max_abs_dt > fast_validation_threshold:
            raise RuntimeError(
                "Optimization with transit_time_method='fast' failed validation: "
                "the final model differs from a Newton-based transit-time evaluation by "
                f"max_abs_dt={max_abs_dt:.2e} d, which exceeds the median timing error "
                f"({fast_validation_threshold:.2e} d). "
                "Consider rerunning the optimization with transit_time_method='newton'. "
                "Once a reliable solution has been found, transit_time_method='fast' can "
                "still be used for subsequent NUTS initialization and sampling."
            )

    if plot:
        tcall = jttv.get_transit_times_all_list(
            pdic_opt,
            transit_orbit_idx=transit_orbit_idx,
        )
        jttv.plot_model(tcall, marker=".", save=save)

    pdic_opt["pmass"] = jnp.exp(pdic_opt["lnpmass"])

    if return_optspace:
        pdic_optspace = params_to_dict(best_popt, npl, keys)
        pdic_optspace["pmass"] = jnp.exp(pdic_optspace["lnpmass"])
        return pdic_opt, pdic_optspace

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
