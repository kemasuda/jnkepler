""" Kepler equation solver based on Markley (1995) """

__all__ = ["get_E"]

import jax.numpy as jnp
from jax import jit, config
config.update('jax_enable_x64', True)


@jit
def get_E1(M, e, alpha):
    d = 3*(1.-e) + alpha*e
    q = 2*alpha*d*(1-e) - M*M
    r = 3*alpha*d*(d-1+e)*M + M*M*M
    w = (jnp.abs(r) + jnp.sqrt(q*q*q + r*r))**(2./3.)
    z = 2*r*w/(w*w+w*q+q*q) + M
    return z/d


@jit
def get_alpha(M, e):
    return (3*jnp.pi*jnp.pi + 1.6*jnp.pi*(jnp.pi-jnp.abs(M))/(1+e))/(jnp.pi*jnp.pi-6.)


@jit
def correct_E(E, M, e):
    ecosE, esinE = e*jnp.cos(E), e*jnp.sin(E)
    f0 = E - esinE - M
    f1 = 1 - ecosE
    f2 = esinE
    f3 = ecosE
    f4 = -esinE
    d3 = -f0 / (f1 - 0.5*f0*f2/f1)
    d4 = -f0 / (f1 + 0.5*d3*f2 + d3*d3*f3/6.)
    d5 = -f0 / (f1 + 0.5*d4*f2 + d4*d4*f3/6. + d4*d4*d4*f4/24.)
    return E + d5


@jit
def get_E(M, e):
    """compute eccentric anomaly given mean anomaly and eccentricity

        Args:
            M: mean anomaly (should be between -pi and pi)
            e: eccentricity

        Returns:
            eccentric anomaly

    """
    alpha = get_alpha(M, e)
    E1 = get_E1(M, e, alpha)
    return correct_E(E1, M, e)
