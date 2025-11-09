__all__ = ["slogdet_jkep_jax", "det_jkep_am",
           "det_jkep_atau", "det_jkep_pm", "det_jkep_ptau"]

from jax import jit, jacrev
from functools import partial
from jnkepler.keplerian import elements_to_xv
from jnkepler.jaxttv.conversion import G
import jax.numpy as jnp


@partial(jit, static_argnums=(1,))
def slogdet_jkep_jax(params, keys):
    """Comouting Jacobian determinant using JAX autodiff"""
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
    """Analytic Jacobian determinant for (a,e,i,Omega,omega,M)"""
    mu, a, e, sini = G * \
        params['mass'], params['a'], params['ecc'], jnp.sin(params['inc'])
    det = 0.5 * mu**(1.5) * a**(0.5) * e * sini
    return det


@jit
def det_jkep_atau(params):
    """Analytic Jacobian determinant for (a,e,i,Omega,omega,tau)"""
    det = det_jkep_am(params)
    n = 2 * jnp.pi / params['period']
    return -n * det


@jit
def det_jkep_pm(params):
    """Analytic Jacobian determinant for (P,e,i,Omega,omega,M)"""
    mu, e, sini = G * params['mass'], params['ecc'], jnp.sin(params['inc'])
    return mu**2 / (6 * jnp.pi) * e * sini


@jit
def det_jkep_ptau(params):
    """Analytic Jacobian determinant for (P,e,i,Omega,omega,tau)"""
    det = det_jkep_pm(params)
    n = 2 * jnp.pi / params['period']
    return -n * det
