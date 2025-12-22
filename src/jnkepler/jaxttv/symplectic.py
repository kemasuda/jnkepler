"""
Symplectic integrator.

JAX-native implementation following the symplectic approach used in TTVFast (https://github.com/kdeck/TTVFast).
"""

__all__ = [
    "integrate_xv", "kepler_step_map", "kick_kepler_map"
]

import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, grad, config, checkpoint, custom_vjp
from jax.lax import scan, while_loop
from .conversion import G
config.update('jax_enable_x64', True)


def dEstep(x, ecosE0, esinE0, dM):
    """single step to solve incremental Kepler's equation to obtain delta(eccentric anomaly)

        Args:
            x: initial guess for dE
            ecosE0, esinE0: eccentricity and eccentric anomaly at the initial state
            dM: delta(mean anomaly)

        Returns:
            dE: updated estimate for delta(eccentric anomaly)

    """
    x2 = x / 2.0  # x = dE
    sx2, cx2 = jnp.sin(x2), jnp.cos(x2)

    sx = 2.0 * sx2 * cx2
    cx = cx2 * cx2 - sx2 * sx2

    f = x + 2.0 * sx2 * (sx2 * esinE0 - cx2 * ecosE0) - dM

    ecosE = cx * ecosE0 - sx * esinE0
    fp = 1.0 - ecosE
    fpp = (sx * ecosE0 + cx * esinE0) / 2.0
    fppp = ecosE / 6.0

    # update (third order)
    dx = -f / fp
    dx = -f / (fp + dx * fpp)
    dx = -f / (fp + dx * (fpp + dx * fppp))

    return x + dx


def _solve_dE_while(ecosE0, esinE0, dM, max_iter, tol):

    def newton_update(dE):
        return dEstep(dE, ecosE0, esinE0, dM)

    i0 = jnp.int32(0)
    dE0 = dM
    err0 = jnp.array(jnp.inf, dtype=dE0.dtype)

    def cond(carry):
        i, dE, err = carry
        return jnp.logical_and(i < max_iter, err > tol)

    def body(carry):
        i, dE, _ = carry
        dE_next = newton_update(dE)
        err_next = jnp.max(jnp.abs(dE_next - dE))
        return (i + 1, dE_next, err_next)

    _, dE, _ = while_loop(cond, body, (i0, dE0, err0))
    return dE


@partial(custom_vjp, nondiff_argnums=(3, 4))
def solve_dE(ecosE0, esinE0, dM, max_iter=10, tol=1e-12):
    """
    Solve for the eccentric-anomaly increment dE in the Kepler step.

    This function solves the scalar equation F(dE)=0 for dE, where
        F(dE) = dE + (1 - cos dE) * (e sin E0) - (sin dE) * (e cos E0) - dM.

    Notes:
        - Uses Newton iterations (up to `max_iter`) with stopping tolerance `tol`.
        - A custom VJP is provided via implicit differentiation of F(dE)=0, so
          gradients do not backpropagate through the Newton iterations.
        - `max_iter` and `tol` are treated as non-differentiable arguments.

    Args:
        ecosE0: e cos(E0), shape (Norbit,).
        esinE0: e sin(E0), shape (Norbit,).
        dM: Mean-anomaly increment n*dt, shape (Norbit,).
        max_iter: Maximum number of Newton iterations.
        tol: Convergence tolerance based on |dE_{k+1} - dE_k|.

    Returns:
        dE: Eccentric-anomaly increment, shape (Norbit,).
    """
    return _solve_dE_while(ecosE0, esinE0, dM, max_iter, tol)


def solve_dE_fwd(ecosE0, esinE0, dM, max_iter=10, tol=1e-12):
    dE = _solve_dE_while(ecosE0, esinE0, dM, max_iter, tol)
    return dE, (dE, ecosE0, esinE0, dM)


def solve_dE_bwd(max_iter, tol, res, dE_bar):
    dE, ecosE0, esinE0, dM = res

    # implicit VJP：F(dE; ecosE0, esinE0, dM)=0
    s, c = jnp.sin(dE), jnp.cos(dE)
    # dF/ddE
    fp = 1.0 + s * esinE0 - c * ecosE0

    # fp = jnp.where(jnp.abs(fp) > 1e-12, fp, jnp.sign(fp) * 1e-12)

    ecosE0_bar = dE_bar * (s / fp)
    esinE0_bar = dE_bar * (-(1.0 - c) / fp)
    dM_bar = dE_bar * (1.0 / fp)
    return (ecosE0_bar, esinE0_bar, dM_bar)


solve_dE.defvjp(solve_dE_fwd, solve_dE_bwd)


def kepler_step(x, v, gm, dt, nitr=10):
    """Kepler step (two-body drift).

        Given Cartesian position/velocity, advance the state by `dt` assuming
        two-body Keplerian motion under the gravitational parameter `gm`.

        Notes:
            The eccentric-anomaly increment ``dE`` is obtained by solving a
            scalar Kepler equation with Newton iterations via `solve_dE`.
            The argument `nitr` is passed as `max_iter` to `solve_dE`, i.e.,
            it sets the maximum number of Newton iterations (not an unrolled
            loop length).

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            gm: 'GM' in Kepler's 3rd law
            dt: time step
            nitr: maximum number of Newton iterations

        Returns:
            tuple:
                - new positions (Norbit, xyz)
                - new velocities (Norbit, xyz)

    """
    r0 = jnp.sqrt(jnp.sum(x * x, axis=1))
    v0s = jnp.sum(v * v, axis=1)
    u = jnp.sum(x * v, axis=1)

    a = 1.0 / (2.0 / r0 - v0s / gm)
    n = jnp.sqrt(gm / (a * a * a))

    ecosE0 = 1.0 - r0 / a
    esinE0 = u / (n * a * a)

    dM = n * dt

    dE = solve_dE(ecosE0, esinE0, dM, max_iter=nitr, tol=1e-12)

    x2 = dE / 2.0
    sx2, cx2 = jnp.sin(x2), jnp.cos(x2)

    sx = 2.0 * sx2 * cx2
    cx = cx2 * cx2 - sx2 * sx2

    f = 1.0 - (a / r0) * (2.0 * sx2 * sx2)
    g = (2.0 * sx2 * (esinE0 * sx2 + cx2 * r0 / a)) / n

    fp = 1.0 - cx * ecosE0 + sx * esinE0
    fdot = -(a / (r0 * fp)) * n * sx
    gdot = (1.0 + g * fdot) / f

    x_new = f[:, None] * x + g[:, None] * v
    v_new = fdot[:, None] * x + gdot[:, None] * v

    return x_new, v_new


def mmat_from_masses(masses):
    """Construct the mass matrix used in the Jacobi-to-astrocentric transform.

    This returns the same lower-triangular mass matrix as used in
    `jacobi_to_astrocentric`. The matrix depends only on the body masses
    and is typically used to convert between coordinate conventions.

    Args:
        masses: Masses of the bodies, shape (Nbody,).

    Returns:
        mmat: Mass matrix, shape (Nbody-1, Nbody-1).

    """
    nbody = len(masses)
    mp = masses[1:]
    return jnp.eye(nbody - 1) + jnp.tril(
        jnp.tile(mp / jnp.cumsum(masses)[1:], (nbody - 1, 1)), k=-1
    )


def Hintgrad(xjac, vjac, masses):
    """gradient of the interaction Hamiltonian times (star mass / planet mass)

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            masses: masses of the bodies (Nbody,), solar unit

        Returns:
            gradient of interaction Hamiltonian x (star mass / planet mass)

    """
    m0 = masses[0]
    mp = masses[1:]              # (N,)
    mu = mp / m0                 # (N,)
    M = mmat_from_masses(masses)  # (N,N)

    # ---- term 1: + sum_i mu_i / |xjac_i|  (Jacobi) ----
    r2 = jnp.sum(xjac * xjac, axis=1)
    r = jnp.sqrt(r2)
    inv_r3 = 1.0 / (r2 * r)
    g_jac = -(mu[:, None] * xjac) * inv_r3[:, None]   # d/dxjac of +sum mu/|x|

    # ---- astrocentric ----
    xast = M @ xjac

    # term 2: - sum_i mu_i / |xast_i|
    r2a = jnp.sum(xast * xast, axis=1)
    ra = jnp.sqrt(r2a)
    inv_r3a = 1.0 / (r2a * ra)
    # d/dxast of (-sum mu/|xast|)
    g_ast_sp = +(mu[:, None] * xast) * inv_r3a[:, None]

    # term 3: -0.5 * sum_{i,j} mu_i mu_j / |xast_i - xast_j|
    diff = xast[:, None, :] - xast[None, :, :]           # (N,N,3)
    d2 = jnp.sum(diff * diff, axis=-1)                   # (N,N)
    nz = d2 != 0.0
    d2_safe = jnp.where(nz, d2, 1.0)
    inv_d3 = jnp.where(nz, 1.0 / (d2_safe * jnp.sqrt(d2_safe)), 0.0)

    w = (mu[:, None] * mu[None, :]) * inv_d3            # (N,N)

    # Because Hint uses -0.5 * sum_{i,j}, the gradient becomes + sum_j mu_i mu_j (x_i-x_j)/r^3
    g_ast_pp = +jnp.sum(w[:, :, None] * diff, axis=1)    # (N,3)

    g_ast = g_ast_sp + g_ast_pp                          # total dHint/dxast

    # chain rule back to Jacobi: xast = M @ xjac  => dH/dxjac += M^T @ dH/dxast
    g_from_ast = M.T @ g_ast

    g = g_jac + g_from_ast

    # correct scaling
    return g * (m0 / mp)[:, None]


def nbody_kicks(x, v, ki, masses, dt):
    """apply N-body kicks to velocities

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            ki: GM values
            masses: masses of the bodies (Nbody,), solar unit
            dt: time step

        Returns:
            tuple:
                - positions
                - kicked velocities

    """
    dv = -ki[:, None] * dt * Hintgrad(x, v, masses)
    return x, v + dv


def integrate_xv(x, v, masses, times, nitr=10):
    """symplectic integration of the orbits

        Args:
            x: initial Jacobi positions (Norbit, xyz)
            v: initial Jacobi velocities (Norbit, xyz)
            masses: masses of the bodies (Nbody,), in units of solar mass
            times: cumulative sum of time steps

        Returns:
            tuple:
                - times (initial time omitted; dt/2 ahead of the input)
                - Jacobi position/velocity array (Nstep, x or v, Norbit, xyz)

    """
    ki = G * masses[0] * jnp.cumsum(masses)[1:] / \
        jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])
    dtarr = jnp.diff(times)

    # transformation between the mapping and real Hamiltonian
    x, v = real_to_mapTO(x, v, ki, masses, dtarr[0])
    # dt/2 ahead of the starting time
    x, v = kepler_step(x, v, ki, dtarr[0] * 0.5, nitr=nitr)

    # advance the system by dt
    def step(xvin, dt):
        x, v = xvin
        x, v = nbody_kicks(x, v, ki, masses, dt)
        xout, vout = kepler_step(x, v, ki, dt, nitr=nitr)
        return [xout, vout], jnp.array([xout, vout])

    step = checkpoint(step)

    _, xv = scan(step, [x, v], dtarr)

    return times[1:] + 0.5 * dtarr[0], xv


def kepler_step_map(xjac, vjac, masses, dt, nitr=10):
    """vmap version of kepler_step; map along the first axis (Ntime)

        Args:
            xjac: Jacobi positions (Ntime, Norbit, xyz)
            vjac: Jacobi velocities (Ntime, Norbit, xyz)
            masses: masses of the bodies (Nbody,), in units of solar mass
            dt: common time step

        Returns:
            new Jacobi positions and velocities (Ntime, x or v, Norbit, xyz)

    """
    ki = G * masses[0] * jnp.cumsum(masses)[1:] / \
        jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])

    def step(x, v): return kepler_step(x, v, ki, dt, nitr=nitr)
    step_map = vmap(step, (0, 0), 0)
    return step_map(xjac, vjac)


def kick_kepler_map(xjac, vjac, masses, dt, nitr=10):
    """vmap version of nbody_kicks + kepler_step; map along the first axis (Ntime)

        Args:
            xjac: jacobi positions (Ntime, Norbit, xyz)
            vjac: jacobi velocities (Ntime, Norbit, xyz)
            masses: masses of the bodies (Nbody,), in units of solar mass
            dt: common time step

        Returns:
            new jacobi positions and velocities (Ntime, x or v, Norbit, xyz)

    """
    ki = G * masses[0] * jnp.cumsum(masses)[1:] / \
        jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])

    def kick_kepler(x, v):
        x, v = nbody_kicks(x, v, ki, masses, 2*dt)
        return kepler_step(x, v, ki, dt, nitr=nitr)
    func_map = vmap(kick_kepler, (0, 0), 0)
    return func_map(xjac, vjac)


def compute_corrector_coefficientsTO():
    """coefficients for the third-order corrector"""
    corr_alpha = jnp.sqrt(7./40.)
    corr_beta = 1. / (48.0 * corr_alpha)

    TOa1, TOa2 = -corr_alpha, corr_alpha
    TOb1, TOb2 = -0.5 * corr_beta, 0.5 * corr_beta

    return TOa1, TOa2, TOb1, TOb2


def corrector_step(x, v, ki, masses, a, b):
    """corrector step

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            ki: GM values
            masses: masses of the bodies (Nbody,), solar unit
            a, b: corrector steps

        Returns:
            new positions and velocities

    """
    _x, _v = kepler_step(x, v, ki, -a)
    _x, _v = nbody_kicks(_x, _v, ki, masses, b)
    _x, _v = kepler_step(_x, _v, ki, a)
    return _x, _v


def real_to_mapTO(x, v, ki, masses, dt):
    """transformation between real and mapping coordinates

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            ki: GM values
            masses: masses of the bodies (Nbody,), solar unit
            dt: time step

        Returns:
            mapped positions and velocities

    """
    TOa1, TOa2, TOb1, TOb2 = compute_corrector_coefficientsTO()
    _x, _v = corrector_step(x, v, ki, masses, TOa2*dt, TOb2*dt)
    _x, _v = corrector_step(_x, _v, ki, masses, TOa1*dt, TOb1*dt)
    return _x, _v
