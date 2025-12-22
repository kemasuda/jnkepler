"""Coordinate transformations and orbital-element utilities."""
__all__ = [
    "reduce_angle", "m_to_u", "tic_to_u", "tic_to_m", "elements_to_xv", "xv_to_elements",
    "jacobi_to_astrocentric", "j2a_map", "astrocentric_to_cm", "a2cm_map", "cm_to_astrocentric",
    "xvjac_to_xvacm", "xvjac_to_xvcm"
]


import jax.numpy as jnp
from jax import jit, vmap, config
from jax.lax import scan
from .markley import get_E
config.update('jax_enable_x64', True)

G = 2.959122082855911e-4


def reduce_angle(M):
    """get angles between -pi and pi

        Args:
            M: angle (radian)

        Returns:
            angle mapped to [-pi, pi)

    """
    return (M + jnp.pi) % (2 * jnp.pi) - jnp.pi


def m_to_u(M, ecc):
    return get_E(reduce_angle(M), ecc)


def tic_to_u(tic, period, ecc, omega, t_epoch):
    """convert time of inferior conjunction to eccentric anomaly u

        Args:
            tic: time of inferior conjunction
            period: orbital period
            ecc: eccentricity
            omega: argument of periastron
            t_epoch: time to which osculating elemetns are referred

        Returns:
            eccentric anomaly at t_epoch

    """
    # tic_to_m
    tanw2 = jnp.tan(0.5 * omega)
    uic = 2 * jnp.arctan(jnp.sqrt((1.-ecc)/(1.+ecc)) *
                         (1.-tanw2)/(1.+tanw2))  # u at t=tic
    M_epoch = 2 * jnp.pi / period * \
        (t_epoch - tic) + uic - ecc * jnp.sin(uic)  # M at t=0
    u_epoch = get_E(reduce_angle(M_epoch), ecc)
    return u_epoch


def tic_to_m(tic, period, ecc, omega, t_epoch):
    """convert time of inferior conjunction to mean anomaly M_epoch

        Args:
            tic: time of inferior conjunction
            period: orbital period
            ecc: eccentricity
            omega: argument of periastron
            t_epoch: time to which osculating elemetns are referred

        Returns:
            mean anomaly at t_epoch

    """
    tanw2 = jnp.tan(0.5 * omega)
    uic = 2 * jnp.arctan(jnp.sqrt((1.-ecc)/(1.+ecc)) *
                         (1.-tanw2)/(1.+tanw2))  # u at t=tic
    M_epoch = 2 * jnp.pi / period * \
        (t_epoch - tic) + uic - ecc * jnp.sin(uic)  # M at t=0
    return M_epoch


def elements_to_xv(porb, ecc, inc, omega, lnode, u, mass):
    """convert single set of orbital elements to position and velocity

        Args:
            porb: orbital period (day)
            ecc: eccentricity
            inc: inclination (radian)
            omega: argument of periastron (radian)
            lnode: longitude of ascending node (radian)
            u: eccentric anomaly (radian)
            mass: mass in Kepler's 3rd law

        Returns:
            tuple:
                - xout: positions (xyz, )
                - vout: velocities (xyz, )

    """
    cosu, sinu = jnp.cos(u), jnp.sin(u)
    cosw, sinw, cosO, sinO, cosi, sini = jnp.cos(omega), jnp.sin(
        omega), jnp.cos(lnode), jnp.sin(lnode), jnp.cos(inc), jnp.sin(inc)

    n = 2 * jnp.pi / porb
    na = (n * G * mass) ** (1./3.)
    R = 1.0 - ecc * cosu

    Pvec = jnp.array([cosw*cosO - sinw*sinO*cosi, cosw *
                     sinO + sinw*cosO*cosi, sinw*sini])
    Qvec = jnp.array([-sinw*cosO - cosw*sinO*cosi, -sinw *
                     sinO + cosw*cosO*cosi, cosw*sini])
    x, y = cosu - ecc, jnp.sqrt(1.-ecc*ecc) * sinu
    vx, vy = -sinu, jnp.sqrt(1.-ecc*ecc) * cosu

    xout = (na / n) * (Pvec * x + Qvec * y)
    vout = (na / R) * (Pvec * vx + Qvec * vy)

    return xout, vout


def xv_to_elements(x, v, ki):
    """convert position/velocity to elements

        Args:
            x, v: position and velocity (Norbit, xyz)
            ki: 'GM' in Kepler's 3rd law (Norbit); depends on what x/v mean (Jacobi, astrocentric, ...)

        Returns:
            array:
                - semi-major axis (au)
                - orbital period (day)
                - eccentricity
                - inclination (radian)
                - argument of periastron (radian)
                - longitude of ascending node (radian)
                - mean anomaly (radian)

    """
    r0 = jnp.sqrt(jnp.sum(x*x, axis=1))
    v0s = jnp.sum(v*v, axis=1)
    u = jnp.sum(x*v, axis=1)
    a = 1. / (2./r0 - v0s/ki)
    n = jnp.sqrt(ki / (a*a*a))
    ecosE0, esinE0 = 1. - r0 / a, u / (n*a*a)
    e = jnp.sqrt(ecosE0**2 + esinE0**2)
    E = jnp.arctan2(esinE0, ecosE0)

    hx = x[:, 1] * v[:, 2] - x[:, 2] * v[:, 1]
    hy = x[:, 2] * v[:, 0] - x[:, 0] * v[:, 2]
    hz = x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0]
    hnorm = jnp.sqrt(hx**2 + hy**2 + hz**2)
    inc = jnp.arccos(hz / hnorm)

    P = (jnp.cos(E) / r0)[:, None] * x - \
        (jnp.sqrt(a / ki) * jnp.sin(E))[:, None] * v
    Q = (jnp.sin(E) / r0 / jnp.sqrt(1 - e*e))[:, None] * x + (
        jnp.sqrt(a / ki) * (jnp.cos(E)-e) / jnp.sqrt(1 - e*e))[:, None] * v
    PQz = jnp.sqrt(P[:, 2]**2 + Q[:, 2]**2)
    omega = jnp.where(PQz != 0., jnp.arctan2(P[:, 2], Q[:, 2]), 0.)
    coslnode = (P[:, 0] * Q[:, 2] - P[:, 2] * Q[:, 0]) / PQz
    sinlnode = (P[:, 1] * Q[:, 2] - P[:, 2] * Q[:, 1]) / PQz
    lnode = jnp.where(PQz != 0., jnp.arctan2(sinlnode, coslnode), 0.)

    return jnp.array([a, 2*jnp.pi/n, e, inc, omega, lnode, E - esinE0])


def jacobi_to_astrocentric(xjac, vjac, masses):
    """conversion from Jacobi to astrocentric

        Args:
            xjac: jacobi positions (Norbit, xyz)
            vjac: jacobi velocities (Norbit, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            tuple:
                - astrocentric positions (Norbit, xyz)
                - astrocentric velocities (Norbit, xyz)

    """
    nbody = len(masses)
    mmat = jnp.eye(
        nbody-1) + jnp.tril(jnp.tile(masses[1:] / jnp.cumsum(masses)[1:], (nbody-1, 1)), k=-1)
    return mmat@xjac, mmat@vjac


# map along the 1st axes of xjac, vjac
j2a_map = vmap(jacobi_to_astrocentric, (0, 0, None), 0)  # not used?


def astrocentric_to_cm(xast, vast, masses):
    """conversion from astrocentric to CoM

        Args:
            xast: astrocentric positions (Norbit, xyz)
            vast: astrocentric velocities (Norbit, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            tuple:
                - CoM positions (Nbody, xyz); now star (index 0) is added
                - CoM velocities (Nbody, xyz); now star (index 0) is added

    """
    mtot = jnp.sum(masses)
    xcm_ast = jnp.sum(masses[1:][:, None] * xast, axis=0) / mtot
    vcm_ast = jnp.sum(masses[1:][:, None] * vast, axis=0) / mtot
    xcm = jnp.vstack([-xcm_ast, xast - xcm_ast])
    vcm = jnp.vstack([-vcm_ast, vast - vcm_ast])
    return xcm, vcm


# map along the 1st axes of xast, vast
a2cm_map = vmap(astrocentric_to_cm, (0, 0, None), 0)


def cm_to_astrocentric(x, v, a, j):
    """astrocentric x/v/a of the jth orbit (planet) from CoM x/v/a

        Args:
            x: CoM positions (Nstep, Nbody, xyz)
            v: CoM velocities (Nstep, Nbody, xyz)
            a: CoM accelerations (Nstep, Nbody, xyz)
            j: orbit (planet) index

        Returns:
            tuple:
                - astrocentric position of jth orbit (Nstep, xyz)
                - astrocentric velocity of jth orbit (Nstep, xyz)
                - astrocentric acceleration of jth orbit (Nstep, xyz)

    """
    xastj = x[:, j, :] - x[:, 0, :]
    vastj = v[:, j, :] - v[:, 0, :]
    aastj = a[:, j, :] - a[:, 0, :]
    return xastj, vastj, aastj


def get_acm(x, masses):
    """compute acceleration given position, velocity, mass

        Args:
            x: positions in CoM frame (Norbit, xyz)
            masses: masses of the bodies (Nbody)

        Returns:
            array: accelerations (Norbit, xyz)

    """
    xjk = jnp.transpose(x[:, None] - x[None, :], axes=[0, 2, 1])
    x2jk = jnp.sum(xjk * xjk, axis=1)[:, None, :]
    x2jk = jnp.where(x2jk != 0., x2jk, jnp.inf)
    x2jkinv = 1. / x2jk

    x2jkinv1p5 = x2jkinv * jnp.sqrt(x2jkinv)
    Xjk = - xjk * x2jkinv1p5

    a = G * jnp.dot(Xjk, masses)

    return a


# map along the 1st axis of x
geta_map = vmap(get_acm, (0, None), 0)


def xvjac_to_xvacm(x, v, masses):
    """Conversion from Jacobi to center-of-mass

        Args:
            xv: positions and velocities in Jacobi coordinates (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,), solar unit

        Returns:
            tuple:
                - xcm: positions in the CoM frame (Nstep, Norbit)
                - vcm: velocities in the CoM frame
                - acm: accelerations in the CoM frame

    """
    xa, va = jacobi_to_astrocentric(x, v, masses)
    xcm, vcm = a2cm_map(xa, va, masses)
    acm = geta_map(xcm, masses)
    return xcm, vcm, acm


def xvjac_to_xvcm(x, v, masses):
    """ Conversion from Jacobi to center-of-mass

        Args:
            xv: positions and velocities in Jacobi coordinates (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,), solar unit

        Returns:
            tuple:
                - xcm: positions in the CoM frame (Nstep, Norbit)
                - vcm: velocities in the CoM frame

    """
    xa, va = jacobi_to_astrocentric(x, v, masses)
    xcm, vcm = a2cm_map(xa, va, masses)
    return xcm, vcm
