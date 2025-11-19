__all__ = ["slogdet_jkep_jax", "det_jkep_am",
           "det_jkep_atau", "det_jkep_pm", "det_jkep_ptau",
           "slogdet_jkep2D_jax", "det_jkep2D_am", "det_jkep2D_pm"]

from jax import jit, jacrev
from functools import partial
from jnkepler.keplerian import elements_to_xv
from jnkepler.jaxttv.conversion import G
import jax.numpy as jnp


@partial(jit, static_argnums=(1,))
def slogdet_jkep_jax(params, keys):
    """
    Compute the Jacobian determinant for the mapping from orbital elements to
    Cartesian state vectors in the 3D Kepler problem using JAX autodiff 

    Args:
        params (dict): Dictionary of orbital-element parameters.
        keys (list[str]): Parameter names with respect to which the Jacobian
            is computed. Only these parameters are differentiated.
            Typical choices correspond to combinations such as
            (a, ecc, inc, lnode, omega, M), (a, ecc, inc, lnode, omega, tau),
            (period, ecc, inc, lnode, omega, M).

    Returns:
        tuple: (sign, log_abs_det) from `jnp.linalg.slogdet`, where the Jacobian
            is taken with respect to the flattened state vector (x, v).
    """
    def func(params):
        if 'a' in keys:
            params['period'] = 2 * jnp.pi * \
                jnp.sqrt(params['a']**3 / G / params['mass'])
        elif 'n' in keys:
            params['period'] = 2 * jnp.pi / params['n']

        if 'M' in keys:
            params['tau'] = params['t_ref'] - \
                params['M'] * params['period'] / 2 / jnp.pi

        out = elements_to_xv(0., params)
        return jnp.hstack([out['x'], out['v']])

    Jdict = jacrev(func)(params)
    Jarr = jnp.stack([Jdict[k].reshape(-1) for k in keys], axis=1)

    return jnp.linalg.slogdet(Jarr)


@jit
def det_jkep_am(params):
    """
    Analytic Jacobian determinant for the transformation
    (a, e, i, Omega, omega, M) → (x, y, z, vx, vy, vz).

    Args:
        params (dict): Orbital-element parameters.

    Returns:
        float: Analytic Jacobian determinant.
    """
    mu, a, e, sini = G * \
        params['mass'], params['a'], params['ecc'], jnp.sin(params['inc'])
    det = 0.5 * mu**(1.5) * a**(0.5) * e * sini
    return det


@jit
def det_jkep_atau(params):
    """
    Analytic Jacobian determinant for the transformation
    (a, e, i, Omega, omega, tau) → (x, y, z, vx, vy, vz).

    Args:
        params (dict): Orbital-element parameters.

    Returns:
        float: Analytic Jacobian determinant.
    """
    det = det_jkep_am(params)
    n = 2 * jnp.pi / params['period']
    return -n * det


@jit
def det_jkep_pm(params):
    """
    Analytic Jacobian determinant for the transformation
    (P, e, i, Omega, omega, M) → (x, y, z, vx, vy, vz).

    Args:
        params (dict): Orbital-element parameters.

    Returns:
        float: Analytic Jacobian determinant.
    """
    mu, e, sini = G * params['mass'], params['ecc'], jnp.sin(params['inc'])
    return mu**2 / (6 * jnp.pi) * e * sini


@jit
def det_jkep_ptau(params):
    """
    Analytic Jacobian determinant for the transformation
    (P, e, i, Omega, omega, tau) → (x, y, z, vx, vy, vz).

    Args:
        params (dict): Orbital-element parameters.

    Returns:
        float: Analytic Jacobian determinant.
    """
    det = det_jkep_pm(params)
    n = 2 * jnp.pi / params['period']
    return -n * det


@partial(jit, static_argnums=(1,))
def slogdet_jkep2D_jax(params, keys):
    """
    Compute the Jacobian determinant for the mapping from orbital elements to
    Cartesian state vectors in the 2D Kepler problem using JAX autodiff 

    Args:
        params (dict): Dictionary of orbital-element parameters.
        keys (list[str]): Parameter names with respect to which the Jacobian
            is computed. Only these parameters are differentiated.
            Typical choices correspond to combinations such as
            (a, ecc, omega, M), (period, ecc, omega, M).

    Returns:
        tuple: (sign, log_abs_det) from `jnp.linalg.slogdet`, where the Jacobian
            is taken with respect to the flattened state vector (x, v).
    """
    def func(params):
        if 'a' in keys:
            params['period'] = 2 * jnp.pi * \
                jnp.sqrt(params['a']**3 / G / params['mass'])
        elif 'n' in keys:
            params['period'] = 2 * jnp.pi / params['n']
        if 'M' in keys:
            params['tau'] = params['t_ref'] - \
                params['M'] * params['period'] / 2 / jnp.pi
        out = elements_to_xv(0., params | {'lnode': 0, 'inc': jnp.pi / 2.})
        return jnp.hstack([out['x'][0][0], out['x'][0][2], out['v'][0][0], out['v'][0][2]])

    exclude = {'lnode', 'inc'}
    params4 = {k: v for k, v in params.items() if k not in exclude}
    Jdict = jacrev(func)(params4)
    Jarr = jnp.stack([Jdict[k].reshape(-1) for k in keys], axis=1)

    return jnp.linalg.slogdet(Jarr)


@jit
def det_jkep2D_am(params):
    """
    Analytic Jacobian determinant for the transformation
    (a, e, omega, M) → (x, z, vx, vz) in the 2D Kepler problem.

    Args:
        params (dict): Orbital-element parameters.

    Returns:
        float: Analytic Jacobian determinant.
    """
    mu, e = G * params['mass'], params['ecc']
    det = 0.5 * mu * e / jnp.sqrt(1. - e**2)
    return det


@jit
def det_jkep2D_pm(params):
    """
    Analytic Jacobian determinant for the transformation
    (P, e, omega, M) → (x, z, vx, vz) in the 2D Kepler problem.

    Args:
        params (dict): Orbital-element parameters.

    Returns:
        float: Analytic Jacobian determinant.
    """
    mu, p, e = G * params['mass'], params['period'], params['ecc']
    a_over_p = (mu / p / 4. / jnp.pi**2)**(1./3.)
    det = (mu / 3.) * e / jnp.sqrt(1. - e**2) * a_over_p
    return det
