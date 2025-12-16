__all__ = ["information_from_model_independent_normal"]

import jax.numpy as jnp
import functools
from collections import OrderedDict
from jax import jacrev, jacfwd, random
from numpyro import handlers, infer
from numpyro.distributions.transforms import biject_to


def _to_unconstrained(model, params_constrained, keys, *model_args, **model_kwargs):
    """
    Convert constrained parameter values to their unconstrained representations
    using the model's sample sites.

    Args:
        model (callable): A NumPyro model function.
        params_constrained (dict): Dictionary mapping parameter names to values
            in the *constrained* space (e.g., positive scales, simplex).
        keys (list): Keys to be included in the output dict.
        *args: Positional arguments passed to `model` when tracing.
        **kwargs: Keyword arguments passed to `model` when tracing.

    Returns:
        dict: Dictionary mapping parameter names to values in the
        *unconstrained* space, suitable for use in inference algorithms
        (e.g., HMC/NUTS).
    """
    tr = handlers.trace(handlers.seed(model, 0)).get_trace(
        *model_args, **model_kwargs)
    bij = {}
    for name, site in tr.items():
        if site["type"] == "sample" and not site["is_observed"]:
            bij[name] = biject_to(site["fn"].support)

    z_params = {k: bij[k].inv(params_constrained[k]) for k in keys}

    return z_params


def _seed_and_substitute(model, params_dict, param_space, rng_key):
    """Seed and substitute parameters into a NumPyro model."""
    if param_space == "unconstrained":
        substituted = handlers.substitute(
            model,
            substitute_fn=functools.partial(
                infer.util._unconstrain_reparam, params_dict),
        )
    elif param_space == "constrained":
        substituted = handlers.substitute(model, data=params_dict)
    else:
        raise ValueError(
            "param_space must be 'constrained' or 'unconstrained'.")
    return handlers.seed(substituted, rng_seed=rng_key)


def _std_residuals_from_model_independent_normal(
    model,
    params_dict,
    param_space,
    rng_key,
    *,
    sigma_sd,
    mu_name="tcmodel",
    obs_name="obs",
    observed=None,
    model_args=(),
    model_kwargs=None,
):
    """
    Build standardized residuals z = (y - mu(theta)) / sigma for independent Gaussian likelihood.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    seeded = _seed_and_substitute(model, params_dict, param_space, rng_key)
    trace = handlers.trace(seeded).get_trace(*model_args, **model_kwargs)

    if mu_name not in trace:
        raise KeyError(
            f"deterministic mu '{mu_name}' not found in trace. "
            "Record it via numpyro.deterministic(mu_name, mu)."
        )
    mu = jnp.asarray(trace[mu_name]["value"]).reshape(-1)
    sigma_sd = jnp.asarray(sigma_sd).reshape(-1)

    if observed is not None:
        y = jnp.asarray(observed).reshape(-1)
    else:
        if (obs_name not in trace):
            raise KeyError(f"obs site '{obs_name}' not found in trace.")
        else:
            y = jnp.asarray(trace[obs_name]["value"]).reshape(-1)

    if y.shape != mu.shape or y.shape != sigma_sd.shape:
        raise ValueError(
            f"shape mismatch: y {y.shape}, mu {mu.shape}, sigma {sigma_sd.shape}")

    return (y - mu) / sigma_sd  # (N,)


def information_from_model_independent_normal(
    *,
    model=None,
    model_args=(),
    model_kwargs=None,
    pdic=None,
    mu_name=None,
    observed=None,
    obs_name=None,
    keys=None,
    sigma_sd=None,
    param_space="unconstrained",
    rng_key=None,
):
    """
    Compute Fisher information matrix for independent Gaussian likelihood directly from a NumPyro model,
    using (observed - mu(pdic)) / sigma_sd obtained from a NumPyro model. 

    Args:
        model: NumPyro model.
        model_args, model_kwargs: static args/kwargs for the model.
        pdic: dict of parameter values in constrained space.
        mu_name: deterministic site name for the model mean.
        observed: 1D array of observed values; obs_name is used if not provided.
        obs_name: observed site name.
        keys: list of parameter names to differentiate (order preserved).
        sigma_sd: 1D array of standard deviations (SD) for iid noise.
        param_space: 'constrained' or 'unconstrained'; use 'unconstrained' to initialize inverse_mass_matrix.
        rng_key: PRNG key (default = jax.random.PRNGKey(0)).

    Returns:
        dict: A dictionary containing the Fisher information results and related metadata:

            - "fisher" (jnp.ndarray): The (P, P) Fisher information matrix.
            - "col_slices" (dict[str, slice]): Mapping from each parameter name to its
              corresponding column range in the Fisher matrix.
            - "col_names" (list[str]): Flattened per-column names, matching the order of
              columns in the Fisher matrix.
            - "params_unconstrained" (dict[str, jnp.ndarray]): Parameter values in the
              unconstrained space used for differentiation.
    """
    assert model is not None and pdic is not None and mu_name is not None and keys is not None and sigma_sd is not None
    if (observed is None) and (obs_name is None):
        raise ValueError("Either `observed` or `obs_name` must be provided.")
    keys = list(keys)

    if param_space == "unconstrained":
        _pdic = _to_unconstrained(
            model, pdic, keys, *model_args, **(model_kwargs or {}))
    elif param_space == "constrained":
        _pdic = dict({k: pdic[k] for k in keys})
    else:
        raise ValueError(
            "param_space must be 'constrained' or 'unconstrained'.")
    pdic_sub = OrderedDict((k, _pdic[k]) for k in keys)
    rng_key = random.PRNGKey(0) if rng_key is None else rng_key
    model_kwargs = {} if model_kwargs is None else model_kwargs
    base = dict({k: v for k, v in _pdic.items()
                if k not in (mu_name, obs_name)})

    def r_fn(p_sub):
        p_all = dict(base)
        p_all.update(p_sub)
        return _std_residuals_from_model_independent_normal(
            model,
            p_all,
            param_space,
            rng_key,
            sigma_sd=sigma_sd,
            mu_name=mu_name,
            obs_name=obs_name,
            model_args=model_args,
            model_kwargs=model_kwargs,
            observed=observed
        )  # (N,)

    # Jacobian of standardized residuals w.r.t. params (ordered by `keys`)
    Jtree = jacrev(r_fn)(pdic_sub)

    # Stack columns in stable key order; flatten trailing dims per key
    N = Jtree[keys[0]].shape[0]
    cols, names, slices, c0 = [], [], {}, 0
    for k in keys:
        Jk = jnp.asarray(Jtree[k]).reshape(N, -1)
        cols.append(Jk)
        d = Jk.shape[1]
        names += [k] if d == 1 else [f"{k}[{i}]" for i in range(d)]
        slices[k] = slice(c0, c0 + d)
        c0 += d
    J = jnp.hstack(cols)   # (N, P)

    F = J.T @ J
    return {
        "fisher": F,
        "col_slices": slices,
        "col_names": names,
        "params_unconstrained": _pdic
    }
