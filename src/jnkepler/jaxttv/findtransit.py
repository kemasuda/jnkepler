"""Routines for finding transit times."""

__all__ = [
    "find_transit_times_all",
    "find_transit_times_fast",
    "find_transit_params_all",
    "find_transit_params_fast",
    "find_transit_times_kepler_all",
]

import jax.numpy as jnp
from jax import checkpoint, config, vmap
from jax.lax import scan

from .conversion import G, jacobi_to_astrocentric, xvjac_to_xvacm
from .hermite4 import hermite4_step_map
from .symplectic import kepler_kick_kepler_map, kepler_step_map, kick_kepler_map
from .utils import find_nearest_idx, find_nearest_idx_sorted

config.update("jax_enable_x64", True)


def _get_tcflag(xjac, vjac):
    """Find steps just after transit centers using Jacobi coordinates.

    Args:
        xjac: Jacobi positions of shape ``(Nstep, Norbit, 3)``.
        vjac: Jacobi velocities of shape ``(Nstep, Norbit, 3)``.

    Returns:
        Boolean array of shape ``(Nstep - 1, Norbit)`` whose entries are
        True at steps just after the transit center.
    """
    g = jnp.sum(xjac[:, :, :2] * vjac[:, :, :2], axis=2)
    return (g[:-1] < 0.0) & (g[1:] > 0.0) & (xjac[1:, :, 2] > 0.0)


def _get_g_map(xjac, vjac, pidxarr):
    """Evaluate ``x · v`` in the sky plane for the selected planets."""

    def g_orbit(xjac_one, vjac_one, j):
        return jnp.sum(xjac_one[j, :2] * vjac_one[j, :2])

    g_map = vmap(g_orbit, (0, 0, 0), 0)
    return g_map(xjac, vjac, pidxarr).ravel()


def _find_tc_idx(t, tcflag, j, tcobs):
    """Find candidate-step indices nearest to the target transit times.

    Args:
        t: Time array of shape ``(Nstep,)``.
        tcflag: Boolean array of shape ``(Nstep - 1, Norbit)`` indicating
            whether each step is just after a transit center.
        j: Orbit index.
        tcobs: Target transit time or times for the selected orbit.

    Returns:
        Indices in ``t[1:]`` closest to ``tcobs`` among the entries where
        ``tcflag[:, j]`` is True.
    """
    tc_candidates = jnp.where(tcflag[:, j], t[1:], -jnp.inf)
    return find_nearest_idx(tc_candidates, jnp.atleast_1d(tcobs))


_find_tc_idx_map = vmap(_find_tc_idx, (None, None, 0, 0), 0)


def _get_nrstep(x, v, a, j):
    """Compute one Newton correction step for the selected planet.

    Args:
        x: Positions in the center-of-mass frame of shape ``(Nbody, 3)``.
        v: Velocities in the center-of-mass frame of shape ``(Nbody, 3)``.
        a: Accelerations in the center-of-mass frame of shape ``(Nbody, 3)``.
        j: Orbit index starting from 0.

    Returns:
        Newton correction step for the selected planet.
    """
    xastj = x[j + 1, :] - x[0, :]
    vastj = v[j + 1, :] - v[0, :]
    aastj = a[j + 1, :] - a[0, :]
    gj = jnp.sum(xastj[:2] * vastj[:2])
    dotgj = jnp.sum(vastj[:2] * vastj[:2]) + jnp.sum(xastj[:2] * aastj[:2])
    return -gj / dotgj


_get_nrstep_map = vmap(_get_nrstep, (0, 0, 0, 0), 0)


def _find_transit_candidates(pidxarr, tcobsarr, t, xvjac):
    """Find integration steps nearest to the target transit times."""
    xjac, vjac = xvjac[:, 0, :, :], xvjac[:, 1, :, :]
    tcflag = _get_tcflag(xjac, vjac)
    return _find_tc_idx_map(t, tcflag, pidxarr, tcobsarr).ravel()


def _prepare_transit_newton_init(tcidx, pidxarr, t, xvjac, masses):
    """Prepare initial states for Newton refinement."""
    tc = t[1:][tcidx]
    xvjac_init = xvjac[1:][tcidx]

    # Bring back the system by dt/2 so that the states are at the
    # conclusions of the symplectic step.
    dt_correct = -0.5 * jnp.diff(t)[0]
    tc += dt_correct
    xjac_init, vjac_init = kepler_step_map(
        xvjac_init[:, 0, :, :], xvjac_init[:, 1, :, :], masses, dt_correct
    )

    xcm_init, vcm_init, acm_init = xvjac_to_xvacm(xjac_init, vjac_init, masses)
    nrstep_init = _get_nrstep_map(xcm_init, vcm_init, acm_init, pidxarr)

    return tc, xcm_init, vcm_init, nrstep_init


def _scan_transit_newton(xcm_init, vcm_init, nrstep_init, pidxarr, masses, nitr):
    """Run Newton refinement for the transit times."""

    def tcstep(xvs, _):
        xin, vin, step = xvs
        xtc, vtc, atc = hermite4_step_map(xin, vin, masses, step)
        xtc = jnp.transpose(xtc, axes=[2, 0, 1])
        vtc = jnp.transpose(vtc, axes=[2, 0, 1])
        atc = jnp.transpose(atc, axes=[2, 0, 1])
        step = _get_nrstep_map(xtc, vtc, atc, pidxarr)
        return (xtc, vtc, step), step

    tcstep = checkpoint(tcstep)
    return scan(tcstep, (xcm_init, vcm_init, nrstep_init), jnp.arange(nitr))


def _find_transit_newton_core(pidxarr, tcobsarr, t, xvjac, masses, nitr):
    """Core Newton-based transit finder."""
    tcidx = _find_transit_candidates(pidxarr, tcobsarr, t, xvjac)
    tc, xcm_init, vcm_init, nrstep_init = _prepare_transit_newton_init(
        tcidx, pidxarr, t, xvjac, masses
    )
    xvs, steps = _scan_transit_newton(
        xcm_init, vcm_init, nrstep_init, pidxarr, masses, nitr
    )
    tc += nrstep_init + jnp.sum(steps, axis=0)
    return tc, xvs


def _find_tc_idx_sorted(t, tcobs):
    """Find indices in ``t[1:]`` whose step-boundary times are nearest to
    the target transit times.

    The times returned by ``integrate_xv`` are at the midpoints of each
    mapping step.  The fast transit finder corrects back by ``dt/2`` to
    obtain step-boundary times before advancing to the observed transits.
    This function searches on the corrected (boundary) times so that the
    nearest step minimises the subsequent advance.

    Args:
        t: Time array of shape ``(Nstep,)``.  Must be uniformly spaced
            (as guaranteed by ``integrate_xv``).
        tcobs: Target transit time or times. A scalar or array-like input is
            accepted.

    Returns:
        Indices in ``t[1:]`` nearest to ``tcobs`` in step-boundary time.
    """
    dt = jnp.diff(t)[0]  # uniform spacing assumed
    return find_nearest_idx_sorted(t[1:] - 0.5 * dt, jnp.atleast_1d(tcobs))


def _advance_to_tcobs_fast(tcidx, pidxarr, tcobsarr, t, xvjac, masses):
    """Advance states to ``tcobsarr`` for the fast transit finder.

    Args:
        tcidx: Indices in ``t[1:]`` nearest to ``tcobsarr``.
        pidxarr: Planet indices corresponding to each transit.
        tcobsarr: Observed transit times.
        t: Time array of shape ``(Nstep,)``.
        xvjac: Jacobi-frame positions and velocities of shape
            ``(Nstep, 2, Norbit, 3)``.
        masses: Mass array of shape ``(Nbody,)``.

    Returns:
        Tuple containing
            - the observed transit times,
            - positions in the center-of-mass frame at ``tcobsarr``,
            - velocities in the center-of-mass frame at ``tcobsarr``, and
            - the transit-time correction evaluated at ``tcobsarr``.
    """
    tc = t[1:][tcidx]
    xvjac_init = xvjac[1:][tcidx]

    # Bring back the system by dt/2 so that the states are at the
    # conclusions of the symplectic step.
    dt_correct = -0.5 * jnp.diff(t)[0]
    tc += dt_correct
    xjac_init, vjac_init = kepler_step_map(
        xvjac_init[:, 0, :, :],
        xvjac_init[:, 1, :, :],
        masses,
        dt_correct,
    )

    # Advance from the step boundary to the observed transit times.
    dt_to_tcobs = tcobsarr - tc
    xjac_tcobs, vjac_tcobs = kepler_kick_kepler_map(
        xjac_init,
        vjac_init,
        masses,
        dt_to_tcobs,
    )

    xcm_tcobs, vcm_tcobs, acm_tcobs = xvjac_to_xvacm(
        xjac_tcobs, vjac_tcobs, masses
    )
    nrstep_tcobs = _get_nrstep_map(xcm_tcobs, vcm_tcobs, acm_tcobs, pidxarr)

    return tcobsarr, xcm_tcobs, vcm_tcobs, nrstep_tcobs


def _find_transit_times_fast_core(pidxarr, tcobsarr, t, xvjac, masses):
    """Core fast transit finder returning only transit times."""
    tcidx = _find_tc_idx_sorted(t, tcobsarr)
    tcobs, _, _, nrstep_tcobs = _advance_to_tcobs_fast(
        tcidx, pidxarr, tcobsarr, t, xvjac, masses
    )
    return tcobs + nrstep_tcobs


def _find_transit_params_fast_core(pidxarr, tcobsarr, t, xvjac, masses):
    """Core fast transit finder returning transit times and phase-space states."""
    tcidx = _find_tc_idx_sorted(t, tcobsarr)
    tcobs, xcm_tcobs, vcm_tcobs, nrstep_tcobs = _advance_to_tcobs_fast(
        tcidx, pidxarr, tcobsarr, t, xvjac, masses
    )
    xcm_tc, vcm_tc, _ = hermite4_step_map(
        xcm_tcobs, vcm_tcobs, masses, nrstep_tcobs)
    xcm_tc = jnp.transpose(xcm_tc, axes=[2, 0, 1])
    vcm_tc = jnp.transpose(vcm_tc, axes=[2, 0, 1])
    return tcobs + nrstep_tcobs, (xcm_tc, vcm_tc, nrstep_tcobs)


def find_transit_times_all(pidxarr, tcobsarr, t, xvjac, masses, nitr=5):
    """Find transit times for all requested transits using Newton refinement.

    Args:
        pidxarr: Orbit indices starting from 0, with shape ``(Ntransit,)``.
        tcobsarr: Flattened array of observed transit times of shape
            ``(Ntransit,)``.
        t: Time array of shape ``(Nstep,)``.
        xvjac: Jacobi positions and velocities of shape
            ``(Nstep, 2, Norbit, 3)``.
        masses: Mass array of shape ``(Nbody,)``.
        nitr: Number of Newton-Raphson iterations.

    Returns:
        Transit times as a one-dimensional array.
    """
    tc, _ = _find_transit_newton_core(
        pidxarr, tcobsarr, t, xvjac, masses, nitr)
    return tc


def find_transit_times_fast(pidxarr, tcobsarr, t, xvjac, masses):
    """Find transit times for all requested transits using the fast algorithm.

    Args:
        pidxarr: Orbit indices starting from 0, with shape ``(Ntransit,)``.
        tcobsarr: Flattened array of observed transit times of shape
            ``(Ntransit,)``.
        t: Time array of shape ``(Nstep,)``.
        xvjac: Jacobi positions and velocities of shape
            ``(Nstep, 2, Norbit, 3)``.
        masses: Mass array of shape ``(Nbody,)``.

    Returns:
        Transit times as a one-dimensional array.
    """
    return _find_transit_times_fast_core(pidxarr, tcobsarr, t, xvjac, masses)


def find_transit_params_all(pidxarr, tcobsarr, t, xvjac, masses, nitr=5):
    """Find transit times and phase-space states using Newton refinement.

    Args:
        pidxarr: Orbit indices starting from 0, with shape ``(Ntransit,)``.
        tcobsarr: Flattened array of observed transit times of shape
            ``(Ntransit,)``.
        t: Time array of shape ``(Nstep,)``.
        xvjac: Jacobi positions and velocities of shape
            ``(Nstep, 2, Norbit, 3)``.
        masses: Mass array of shape ``(Nbody,)``.
        nitr: Number of Newton-Raphson iterations.

    Returns:
        Tuple containing
            - transit times as a one-dimensional array, and
            - positions, velocities, and Newton steps from the final iteration.
    """
    return _find_transit_newton_core(pidxarr, tcobsarr, t, xvjac, masses, nitr)


def find_transit_params_fast(pidxarr, tcobsarr, t, xvjac, masses):
    """Find transit times and phase-space states using the fast algorithm.

    Args:
        pidxarr: Orbit indices starting from 0, with shape ``(Ntransit,)``.
        tcobsarr: Flattened array of observed transit times of shape
            ``(Ntransit,)``.
        t: Time array of shape ``(Nstep,)``.
        xvjac: Jacobi positions and velocities of shape
            ``(Nstep, 2, Norbit, 3)``.
        masses: Mass array of shape ``(Nbody,)``.

    Returns:
        Tuple containing
            - transit times as a one-dimensional array, and
            - positions, velocities, and the time corrections evaluated at
              ``tcobsarr``.
    """
    return _find_transit_params_fast_core(pidxarr, tcobsarr, t, xvjac, masses)


"""TTVFast algorithm."""


def _get_elements(x, v, gm):
    """Compute orbital quantities used by the TTVFast interpolation step.

    Args:
        x: Positions of shape ``(Norbit, 3)``.
        v: Velocities of shape ``(Norbit, 3)``.
        gm: ``GM`` for each orbit.

    Returns:
        Tuple containing the mean motion, ``e cos E0``, ``e sin E0``, and
        ``a / r0``.
    """
    r0 = jnp.sqrt(jnp.sum(x * x, axis=1))
    v0s = jnp.sum(v * v, axis=1)
    u = jnp.sum(x * v, axis=1)
    a = 1.0 / (2.0 / r0 - v0s / gm)

    n = jnp.sqrt(gm / (a * a * a))
    ecosE0, esinE0 = 1.0 - r0 / a, u / (n * a * a)

    return n, ecosE0, esinE0, a / r0


def _find_transit_times_kepler(xast, vast, kast, dt, nitr):
    """Find transit times via the TTVFast interpolation scheme.

    Note:
        This function is adapted from TTVFast
        https://github.com/kdeck/TTVFast, based on the scheme developed by
        Nesvorny et al. (2013, ApJ, 777, 3).

    Args:
        xast: Astrocentric positions of shape ``(Norbit, 3)``.
        vast: Astrocentric velocities of shape ``(Norbit, 3)``.
        kast: Astrocentric ``GM``.
        dt: Integration time step.
        nitr: Number of iterations used in the interpolation solve.

    Returns:
        Time to the transit center.
    """
    n, ecosE0, esinE0, a_r0 = _get_elements(xast, vast, kast)
    rsquared = jnp.sum(xast[:, :2] * xast[:, :2], axis=1)
    vsquared = jnp.sum(vast[:, :2] * vast[:, :2], axis=1)
    xdotv = jnp.sum(xast[:, :2] * vast[:, :2], axis=1)

    def dEstep_transit(dE, _):
        x2 = dE / 2.0
        sx2, cx2 = jnp.sin(x2), jnp.cos(x2)
        f = 1.0 - a_r0 * 2.0 * sx2 * sx2
        sx, cx = 2.0 * sx2 * cx2, cx2 * cx2 - sx2 * sx2
        g = (2.0 * sx2 * (esinE0 * sx2 + cx2 / a_r0)) / n
        fp = 1.0 - cx * ecosE0 + sx * esinE0
        fdot = -(a_r0 / fp) * n * sx
        fp2 = sx * ecosE0 + cx * esinE0
        gdot = 1.0 - 2.0 * sx2 * sx2 / fp

        dgdotdz = -sx / fp + 2.0 * sx2 * sx2 / fp / fp * fp2
        dfdz = -a_r0 * sx
        dgdz = (sx * esinE0 - (ecosE0 - 1.0) * cx) / n
        dfdotdz = -n * a_r0 / fp * (cx + sx / fp * fp2)

        dotproduct = (
            f * fdot * rsquared
            + g * gdot * vsquared
            + (f * gdot + g * fdot) * xdotv
        )
        dotproductderiv = (
            dfdz * (gdot * xdotv + fdot * rsquared)
            + dfdotdz * (f * rsquared + g * xdotv)
            + dgdz * (fdot * xdotv + gdot * vsquared)
            + dgdotdz * (g * vsquared + f * xdotv)
        )

        return dE - dotproduct / dotproductderiv, None

    dE0 = n * dt / 2.0
    dE, _ = scan(dEstep_transit, dE0, jnp.arange(nitr))
    x2 = dE / 2.0
    sx2, cx2 = jnp.sin(x2), jnp.cos(x2)
    sx = 2.0 * sx2 * cx2
    transitM = dE + esinE0 * 2.0 * sx2 * sx2 - sx * ecosE0

    return transitM / n


_find_transit_times_kepler_map = vmap(
    _find_transit_times_kepler, (0, 0, 0, None, None), 0
)


def find_transit_times_kepler_all(pidxarr, tcobsarr, t, xvjac, masses, nitr=3):
    """Find transit times via the legacy TTVFast-style interpolation scheme.

    Note:
        This function is kept for backward compatibility and will be deprecated. 
        It may fail for large ``dt``.

    Args:
        pidxarr: Orbit indices starting from 0, with shape ``(Ntransit,)``.
        tcobsarr: Flattened array of observed transit times of shape
            ``(Ntransit,)``.
        t: Time array of shape ``(Nstep,)``.
        xvjac: Jacobi positions and velocities of shape
            ``(Nstep, 2, Norbit, 3)``.
        masses: Mass array of shape ``(Nbody,)``.
        nitr: Number of iterations used in the Kepler interpolation step.

    Returns:
        Transit times as a one-dimensional array.
    """
    xjac, vjac = xvjac[:, 0, :, :], xvjac[:, 1, :, :]
    tcflag = _get_tcflag(xjac, vjac)
    tcidx = _find_tc_idx_map(t, tcflag, pidxarr, tcobsarr).ravel()

    tc_ahead, tc_behind = t[1:][tcidx], t[1:][tcidx - 1]
    xvjac_ahead, xvjac_behind = xvjac[1:][tcidx], xvjac[1:][tcidx - 1]

    # Bring back the system by dt/2 so that the states are at the
    # conclusions of the symplectic step. If the transit is not bracketed
    # after this shift, advance the system by dt again.
    dt = jnp.diff(t)[0]
    dt2 = 0.5 * dt
    xjac_ahead_mindt2, vjac_ahead_mindt2 = kepler_step_map(
        xvjac_ahead[:, 0, :, :], xvjac_ahead[:, 1, :, :], masses, -dt2
    )
    xjac_behind_mindt2, vjac_behind_mindt2 = kepler_step_map(
        xvjac_behind[:, 0, :, :], xvjac_behind[:, 1, :, :], masses, -dt2
    )
    xjac_ahead_plusdt2, vjac_ahead_plusdt2 = kick_kepler_map(
        xvjac_ahead[:, 0, :, :], xvjac_ahead[:, 1, :, :], masses, dt2
    )
    tcflag_mindt2 = _get_g_map(
        xjac_ahead_mindt2, vjac_ahead_mindt2, pidxarr) > 0.0

    def _select(mask, left, right):
        return jnp.where(mask, left, right)

    select_map = vmap(_select, (0, 0, 0), 0)
    xjac_ahead = select_map(
        tcflag_mindt2, xjac_ahead_mindt2, xjac_ahead_plusdt2)
    xjac_behind = select_map(
        tcflag_mindt2, xjac_behind_mindt2, xjac_ahead_mindt2)
    vjac_ahead = select_map(
        tcflag_mindt2, vjac_ahead_mindt2, vjac_ahead_plusdt2)
    vjac_behind = select_map(
        tcflag_mindt2, vjac_behind_mindt2, vjac_ahead_mindt2)
    tc_ahead = jnp.where(tcflag_mindt2, tc_ahead - dt2, tc_ahead + dt2)
    tc_behind = jnp.where(tcflag_mindt2, tc_behind - dt2, tc_behind + dt2)

    xast_ahead, vast_ahead = jacobi_to_astrocentric(
        xjac_ahead, vjac_ahead, masses)
    xast_behind, vast_behind = jacobi_to_astrocentric(
        xjac_behind, vjac_behind, masses)

    kast = G * (masses[1:] + masses[0])
    kastarr = kast[pidxarr]

    tau_ahead = tc_ahead + jnp.diag(
        _find_transit_times_kepler_map(
            xast_ahead, vast_ahead, kastarr, -dt, nitr)[:, pidxarr]
    )
    tau_behind = tc_behind + jnp.diag(
        _find_transit_times_kepler_map(
            xast_behind, vast_behind, kastarr, dt, nitr)[:, pidxarr]
    )

    tc = (
        (tau_behind - tc_behind) * tau_ahead
        + (tc_ahead - tau_ahead) * tau_behind
    ) / (dt + tau_behind - tau_ahead)

    return tc
