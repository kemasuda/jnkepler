__all__ = ["tic_to_tau", "radial_velocity_shape", "radial_velocity_shape_multi",
           "elements_to_xv", "elements_to_xv_scaled", "xv_to_elements"]

import jax.numpy as jnp
from jax import vmap
from ..jaxttv.conversion import m_to_u
from ..jaxttv.conversion import G
from ..jaxttv.conversion import elements_to_xv as _elements_to_xv
from ..jaxttv.conversion import xv_to_elements as _xv_to_elements


def tic_to_tau(tic, period, ecc, omega):
    """Compute time of periastron passage from time of inferior conjunction tic

        Args:
            tic: time of inferior conjunction where omega + f = pi/2
            period: orbital period
            ecc: eccentricity
            omega: argument of periastron

        Returns:
            float: time of periastron passage

    """
    tanw2 = jnp.tan(0.5 * omega)
    u0 = 2 * jnp.arctan(jnp.sqrt((1. - ecc)/(1. + ecc))
                        * (1. - tanw2)/(1. + tanw2))
    tau = tic - period / (2. * jnp.pi) * (u0 - ecc * jnp.sin(u0))
    return tau


def radial_velocity_shape(t, params):
    """Compute cos(omega+f) + e*cos(omega)

        Args:
            t: times at which RVs are computed
            porb: period
            ecc: eccentricity
            omega: argument of periastron
            tau: time of periastron passage

        Returns:
            array: radial velocities

    """
    M = 2 * jnp.pi * (t - params['tau']) / params['period']
    e, omega = params['ecc'], params['omega']
    u = m_to_u(M, e)
    f = 2 * jnp.arctan(jnp.sqrt((1. + e)/(1. - e)) * jnp.tan(0.5 * u))
    vz_unit_amp = jnp.cos(omega + f) + e * jnp.cos(omega)
    return vz_unit_amp


def radial_velocity_shape_multi(t, params):
    """Vectorized over leading axis of each leaf in params_all (dict-of-arrays)."""
    return vmap(lambda p: radial_velocity_shape(t, p))(params)


def elements_to_xv(t, params):
    """Convert orbital elements to Cartesian state vectors.

        Args:
            t (array_like):
                Times (days) at which positions and velocities are evaluated.
            params (dict):
                Dictionary containing per-orbit orbital elements:
                    - period : orbital period (days)
                    - ecc : eccentricity
                    - inc : inclination (radian)
                    - omega : argument of periastron (radian)
                    - lnode : longitude of ascending node (radian)
                    - tau : time of periastron passage (days)
                    - mass : total mass (solar masses)

        Returns:
            dict: 
                Cartesian position and velocity vectors:
                    - x : array of shape (T, N, 3) if multiple orbits, or (T, 3) if a single orbit. Units: AU.
                    - v : array of shape (T, N, 3) or (T, 3). Units: AU/day.
    """
    t = jnp.atleast_1d(t)
    porb = jnp.atleast_1d(params['period'])
    ecc = jnp.atleast_1d(params['ecc'])
    inc = jnp.atleast_1d(params['inc'])
    omega = jnp.atleast_1d(params['omega'])
    lnode = jnp.atleast_1d(params['lnode'])
    tau = jnp.atleast_1d(params['tau'])
    mass = jnp.atleast_1d(params['mass'])

    # If mass is scalar but you have multiple orbits, repeat it (optional but safe)
    N = porb.size
    if mass.size == 1 and N > 1:
        mass = jnp.repeat(mass, N)

    mean_anom = 2 * jnp.pi * (t[:, None] - tau) / porb
    eccentric_anom = m_to_u(mean_anom, ecc)
    _elements_to_xv_vmap = vmap(
        _elements_to_xv, (None, None, None, None, None, 0, None), 0)
    x, v = _elements_to_xv_vmap(
        porb, ecc, inc, omega, lnode, eccentric_anom, mass)
    x = jnp.swapaxes(x, 1, 2)  # -> (T, N, 3)
    v = jnp.swapaxes(v, 1, 2)
    if x.shape[1] == 1:
        x = x[:, 0]
        v = v[:, 0]
    return dict(x=x, v=v)


def elements_to_xv_scaled(t, params):
    """Convert orbital elements to state vectors scaled by semi-major axis a.

        Args:
            t (array_like): 
                Times (days) at which positions and velocities are evaluated.
            params (dict):
                Dictionary containing per-orbit orbital elements:
                    - period : orbital period (days)
                    - ecc : eccentricity
                    - inc : inclination (radian)
                    - omega : argument of periastron (radian)
                    - lnode : longitude of ascending node (radian)
                    - tau : time of periastron passage (days)

        Returns:
            dict: 
                Cartesian position and velocity vectors divided by a:
                    - x : array of shape (T, N, 3) or (T, 3). Dimensionless.
                    - v : array of shape (T, N, 3) or (T, 3). Units: 1/day.
    """
    par = params.copy()
    par['mass'] = (2 * jnp.pi / params['period'])**2 / \
        G  # set mass so that a=1
    return elements_to_xv(t, par)


def xv_to_elements(x, v, mass, t_ref=None):
    """Convert Cartesian state vectors to orbital elements.

    Args:
        x : array_like
            Cartesian position vector(s) in AU, shape (3,) or (N, 3).
        v : array_like
            Cartesian velocity vector(s) in AU/day, shape (3,) or (N, 3).
        mass : float or array_like
            Total mass (solar masses).
        t_ref : float or array_like, optional
            Reference epoch (days). If provided, the time of periastron
            passage (tau) is computed as tau = t_ref - M/n.

    Returns:
        dict
            Orbital elements:
                - a : semi-major axis (AU)
                - period : orbital period (days)
                - ecc : eccentricity
                - inc : inclination (radian)
                - omega : argument of periastron (radian)
                - lnode : longitude of ascending node (radian)
                - M : mean anomaly at t_ref (radian)
                - tau : time of periastron passage (days), if t_ref is given
                - mass : total mass (solar masses)

            Each value is a scalar if a single orbit is given,
            or a 1-D array of length N for multiple orbits.
    """
    GM = jnp.atleast_1d(G * mass)
    x = jnp.atleast_2d(x)
    v = jnp.atleast_2d(v)
    elements = _xv_to_elements(x, v, GM)
    keys = ['a', 'period', 'ecc', 'inc', 'omega', 'lnode', 'M']
    out = dict(zip(keys, elements))
    out['mass'] = mass
    if t_ref is not None:
        n = 2.0 * jnp.pi / out['period']
        out['tau'] = t_ref - out['M'] / n
    if x.shape[0] == 1:
        out = {k: jnp.squeeze(v) for k, v in out.items()}
    return out
