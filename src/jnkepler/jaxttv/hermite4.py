""" 4th-order Hermite integrator based on Kokubo, E., & Makino, J. 2004, PASJ, 56, 861
used for transit time computation
"""
__all__ = [
    "hermite4_step_map", "integrate_xv",
]


import jax.numpy as jnp
from jax import jit, vmap, grad, config
from jax.lax import scan
from .conversion import BIG_G
config.update('jax_enable_x64', True)


def get_derivs(x, v, masses):
    """compute acceleration and jerk given position, velocity, mass

        Args:
            x: positions in CoM frame (Norbit, xyz)
            v: velocities in CoM frame (Norbit, xyz)
            masses: masses of the bodies (Nbody)

        Returns:
            tuple:
                - accelerations (Norbit, xyz)
                - time derivatives of accelerations (Norbit, xyz)

    """
    xjk = jnp.transpose(x[:, None] - x[None, :], axes=[0, 2, 1])
    vjk = jnp.transpose(v[:, None] - v[None, :], axes=[0, 2, 1])
    x2jk = jnp.sum(xjk * xjk, axis=1)[:, None, :]
    xvjk = jnp.sum(xjk * vjk, axis=1)[:, None, :]

    x2jk = jnp.where(x2jk != 0., x2jk, jnp.inf)
    x2jkinv = 1. / x2jk
    x2jkinv1p5 = x2jkinv * jnp.sqrt(x2jkinv)
    Xjk = - xjk * x2jkinv1p5
    dXjk = (- vjk + 3 * xvjk * xjk * x2jkinv) * x2jkinv1p5

    a = BIG_G * jnp.dot(Xjk, masses)
    adot = BIG_G * jnp.dot(dXjk, masses)

    return a, adot


def predict(x, v, a, dota, dt):
    """predictor step of Hermite integration

        Args:
            x: positions in CoM frame (Norbit, xyz)
            v: velocities in CoM frame (Norbit, xyz)
            a: accelerations in CoM frame (Norbit, xyz)
            adot: jerks in CoM frame (Norbit, xyz)
            dt: time step

        Returns:
            tuple:
                - new positions
                - new velocities

    """
    xp = x + dt * (v + 0.5 * dt * (a + dt * dota / 3.))
    vp = v + dt * (a + 0.5 * dt * dota)
    return xp, vp


def correct(xp, vp, a1, dota1, a, dota, dt, alpha=7./6.):
    """corrector step of Hermite integration

        Args:
            xp: positions in CoM frame (Norbit, xyz), predictor
            vp: velocities in CoM frame (Norbit, xyz), predictor
            a1: accelerations in CoM frame (Norbit, xyz), original state
            adot: jerks in CoM frame (Norbit, xyz), original state
            a1: accelerations in CoM frame (Norbit, xyz) from predictor
            adot1: jerks in CoM frame (Norbit, xyz) from predictor
            dt: time step

        Returns:
            tuple:
                - corrected positions
                - corrected velocities

    """
    a02 = (-6 * (a - a1) - 2 * dt * (2 * dota + dota1)) / dt**2
    a03 = (12 * (a - a1) + 6 * dt * (dota + dota1)) / dt**3
    xc = xp + (dt**4 / 24.) * (a02 + alpha * a03 * dt / 5.)
    vc = vp + (dt**3 / 6.) * (a02 + a03 * dt / 4.)
    return xc, vc


def hermite4_step(x, v, masses, dt):
    """advance the system by a single predictor-corrector step

        Args:
            x: positions in CoM frame (Norbit, xyz)
            v: velocities in CoM frame (Norbit, xyz)
            masses: masses of the bodies (Nbody)
            dt: timestep

        Returns:
            new positions, new velocities, 'new' accelerations

    """
    a, dota = get_derivs(x, v, masses)
    xp, vp = predict(x, v, a, dota, dt)
    a1, dota1 = get_derivs(xp, vp, masses)
    xc, vc = correct(xp, vp, a1, dota1, a, dota, dt)
    return xc, vc, a1


# map along the 1st axes of x, v, dt (Ntransits)
# xva, body idx, xyz, transit idx
hermite4_step_map = jit(vmap(hermite4_step, (0, 0, None, 0), 2))


def integrate_xv(x, v, masses, times):
    """Hermite integration of the orbits

        Args:
            x: initial CoM positions (Norbit, xyz)
            v: initial CoM velocities (Norbit, xyz)
            masses: masses of the bodies (Nbody,), in units of solar mass
            times: cumulative sum of time steps

        Returns:
            tuple:
                - times (initial time omitted)
                - CoM position/velocity array (Nstep, x or v, Norbit, xyz)

    """
    dtarr = jnp.diff(times)

    def step(xvin, dt):
        xin, vin = xvin
        xout, vout, a1 = hermite4_step(xin, vin, masses, dt)
        return [xout, vout], jnp.array([xout, vout, a1])

    _, xv = scan(step, [x, v], dtarr)

    return times[1:], xv


"""
def integrate_elements(elements, masses, times, t_epoch):
    integration given elements and masses
    xrel_j, vrel_j = initialize_from_elements(elements, masses, t_epoch)
    xrel_ast, vrel_ast = jacobi_to_astrocentric(xrel_j, vrel_j, masses)
    x, v = astrocentric_to_cm(xrel_ast, vrel_ast, masses)
    t, xva = integrate_xv(x, v, masses, times)
    return t, xva
"""
