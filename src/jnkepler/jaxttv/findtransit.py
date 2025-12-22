"""Routines for finding transit times.
"""
__all__ = [
    "find_transit_times_single", "find_transit_times_all", "find_transit_params_all",
    "find_transit_times_kepler_all"
]

import jax.numpy as jnp
from jax import jit, vmap, grad, config, checkpoint
from jax.lax import scan
from functools import partial
from .conversion import cm_to_astrocentric, xvjac_to_xvacm, jacobi_to_astrocentric, G
from .symplectic import kepler_step_map, kick_kepler_map
from .hermite4 import hermite4_step_map
from .utils import findidx_map, get_energy_map
config.update('jax_enable_x64', True)


def get_gderivs(xastj, vastj, aastj):
    """time derivatives of g=x^2+y^2 function (squared sky-projected star-planet distance)

        Args:
            xastj: astrocentric positions (Norbit, xyz)
            vastj: astrocentric velocities (Norbit, xyz)
            aastj: astrocentric accelerations (Norbit, xyz)

        Returns:
            values of g, dg/dt (Norbit,)

    """
    gj = jnp.sum(xastj[:, :2] * vastj[:, :2], axis=1)
    dotgj = jnp.sum(vastj[:, :2] * vastj[:, :2], axis=1) + \
        jnp.sum(xastj[:, :2] * aastj[:, :2], axis=1)
    return gj, dotgj


def find_transit_times_single(t, x, v, a, j, masses, nitr=5):
    """find transit times (cannot be jitted)

        Args:
            t: times (Nstep,)
            x: positions in CoM frame (Nstep, Norbit, xyz)
            v: velocities in CoM frame (Nstep, Norbit, xyz)
            a: accelerations in CoM frame (Nstep, Norbit, xyz)
            j: index of the orbit (planet) for each transit times are computed
            masses: masses of the bodies (Nbody,), solar unit
            niter: number of Newton-Raphson iterations

        Returns:
            transit times for the jth orbit (planet) during integration

    """
    xastj, vastj, aastj = cm_to_astrocentric(x, v, a, j)
    gj, dotgj = get_gderivs(xastj, vastj, aastj)

    # step after the sign was changed
    tcidx = (gj[1:] * gj[:-1] < 0) & (xastj[1:, 2] > 0) & (dotgj[1:] > 0)
    tc = t[1:][tcidx]

    nrstep = - (gj / dotgj)[1:][tcidx]
    xtc = x[1:, :, :][tcidx]
    vtc = v[1:, :, :][tcidx]

    for i in range(nitr):
        tc += nrstep
        xtc, vtc, atc = hermite4_step_map(xtc, vtc, masses, nrstep)
        xtc = jnp.transpose(xtc, axes=[2, 0, 1])
        vtc = jnp.transpose(vtc, axes=[2, 0, 1])
        atc = jnp.transpose(atc, axes=[2, 0, 1])
        _xastj, _vastj, _aastj = cm_to_astrocentric(xtc, vtc, atc, j)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        nrstep = - _gj / _dotgj

    return tc


""" Newton-Raphson method w/o for loop """


def get_tcflag(xjac, vjac):
    """find times just after the transit centers using *Jacobi* coordinates

        Args:
            xjac: jacobi positions (Nstep, Norbit, xyz)
            vjac: jacobi velocities (Nstep, Norbit, xyz)

        Returns:
            array (bool): True if the time is just after the transit center (Nstep-1,)

    """
    g = jnp.sum(xjac[:, :, :2] * vjac[:, :, :2], axis=2)  # Nstep, Norbit
    tcflag = (g[:-1] < 0) & (g[1:] > 0) & (xjac[1:, :, 2] > 0)
    return tcflag


def get_g_map(xjac, vjac, pidxarr):
    def g_orbit(xjac, vjac, j):
        return jnp.sum(xjac[j, :2] * vjac[j, :2])
    g_map = vmap(g_orbit, (0, 0, 0), 0)
    return g_map(xjac, vjac, pidxarr).ravel()


def find_tc_idx(t, tcflag, j, tcobs):
    """find indices for times where tcflag is True

        Args:
            t: times (Nstep,)
            tcflag: True if the time is just after the transit center (Nstep-1,)
            j: orbit (planet) index
            tcobs: transit times for jth orbit (planet)

        Returns:
            array: indices of times cloeset to transit centers (Nstep-1,); should be put into times[1:], x[1:], etc.

    """
    tc_candidates = jnp.where(tcflag[:, j], t[1:], -jnp.inf)
    tcidx = findidx_map(tc_candidates, jnp.atleast_1d(tcobs))
    return tcidx


# map along the transit axis
find_tc_idx_map = vmap(find_tc_idx, (None, None, 0, 0), 0)


def get_nrstep(x, v, a, j):
    """compute NR step for jth orbit (planet)

        Args:
            x: positions in CM frame (Norbit, xyz)
            v: velocities in CM frame (Norbit, xyz)
            a: accels in CM frame (Norbit, xyz)
            j: orbit (planet) index, starting from 0

        Returns:
            NR step for jth orbit (planet)

    """
    xastj = x[j+1, :] - x[0, :]
    vastj = v[j+1, :] - v[0, :]
    aastj = a[j+1, :] - a[0, :]
    gj = jnp.sum(xastj[:2] * vastj[:2])
    dotgj = jnp.sum(vastj[:2] * vastj[:2]) + jnp.sum(xastj[:2] * aastj[:2])
    stepj = - gj / dotgj
    return stepj


# map along the transit axis
get_nrstep_map = vmap(get_nrstep, (0, 0, 0, 0), 0)


def find_transit_times_all(pidxarr, tcobsarr, t, xvjac, masses, nitr=5):
    """find transit times for all planets

        Args:
            pidxarr: array of orbit index starting from 0 (Ntransit,)
            tcobsarray: flattened array of observed transit times (Ntransit,)
            t: times (Nstep,)
            xvjac: Jacobi positions and velocities (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,)
            nitr: number of Newton-Raphson iterations

        Returns:
            transit times (1D flattened array)

    """
    xjac, vjac = xvjac[:, 0, :, :], xvjac[:, 1, :, :]
    tcflag = get_tcflag(xjac, vjac)
    tcidx = find_tc_idx_map(t, tcflag, pidxarr, tcobsarr).ravel()

    tc = t[1:][tcidx]
    xvjac_init = xvjac[1:][tcidx]

    # bring back the system by dt/2 so that the systems are at conclusions of the symplectic step
    dt_correct = -0.5 * jnp.diff(t)[0]
    tc += dt_correct
    xjac_init, vjac_init = kepler_step_map(
        xvjac_init[:, 0, :, :], xvjac_init[:, 1, :, :], masses, dt_correct)

    xcm_init, vcm_init, acm_init = xvjac_to_xvacm(xjac_init, vjac_init, masses)
    nrstep_init = get_nrstep_map(xcm_init, vcm_init, acm_init, pidxarr)

    def tcstep(xvs, i):
        xin, vin, step = xvs
        xtc, vtc, atc = hermite4_step_map(xin, vin, masses, step)
        xtc = jnp.transpose(xtc, axes=[2, 0, 1])
        vtc = jnp.transpose(vtc, axes=[2, 0, 1])
        atc = jnp.transpose(atc, axes=[2, 0, 1])
        step = get_nrstep_map(xtc, vtc, atc, pidxarr)
        return [xtc, vtc, step], step

    tcstep = checkpoint(tcstep)

    _, steps = scan(tcstep, [xcm_init, vcm_init,
                    nrstep_init], jnp.arange(nitr))
    tc += nrstep_init + jnp.sum(steps, axis=0)

    return tc


def find_transit_params_all(pidxarr, tcobsarr, t, xvjac, masses, nitr=5):
    """find transit times for all planets

        Args:
            pidxarr: array of orbit index starting from 0 (Ntransit,)
            tcobsarray: flattened array of observed transit times (Ntransit,)
            t: times (Nstep,)
            xvjac: Jacobi positions and velocities (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,)
            nitr: number of Newton-Raphson iterations

        Returns:
            transit times (1D flattened array)

    """
    xjac, vjac = xvjac[:, 0, :, :], xvjac[:, 1, :, :]
    tcflag = get_tcflag(xjac, vjac)
    tcidx = find_tc_idx_map(t, tcflag, pidxarr, tcobsarr).ravel()

    tc = t[1:][tcidx]
    xvjac_init = xvjac[1:][tcidx]

    # bring back the system by dt/2 so that the systems are at conclusions of the symplectic step
    dt_correct = -0.5 * jnp.diff(t)[0]
    tc += dt_correct
    xjac_init, vjac_init = kepler_step_map(
        xvjac_init[:, 0, :, :], xvjac_init[:, 1, :, :], masses, dt_correct)

    xcm_init, vcm_init, acm_init = xvjac_to_xvacm(xjac_init, vjac_init, masses)
    nrstep_init = get_nrstep_map(xcm_init, vcm_init, acm_init, pidxarr)

    def tcstep(xvs, i):
        xin, vin, step = xvs
        xtc, vtc, atc = hermite4_step_map(xin, vin, masses, step)
        xtc = jnp.transpose(xtc, axes=[2, 0, 1])
        vtc = jnp.transpose(vtc, axes=[2, 0, 1])
        atc = jnp.transpose(atc, axes=[2, 0, 1])
        step = get_nrstep_map(xtc, vtc, atc, pidxarr)
        return [xtc, vtc, step], step

    tcstep = checkpoint(tcstep)

    xvs, steps = scan(tcstep, [xcm_init, vcm_init,
                      nrstep_init], jnp.arange(nitr))
    tc += nrstep_init + jnp.sum(steps, axis=0)

    return tc, xvs


""" TTVFast algorithm """


def get_elements(x, v, gm):
    """get elements

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            gm: 'GM' in Kepler's 3rd law

        Returns:
            tuple:
                - n: mean motion
                - ecosE0, esinE0: eccentricity and eccentric anomaly
                - a/r0: semi-major axis divided by |x|

    """
    r0 = jnp.sqrt(jnp.sum(x*x, axis=1))
    v0s = jnp.sum(v*v, axis=1)
    u = jnp.sum(x*v, axis=1)
    a = 1. / (2./r0 - v0s/gm)

    n = jnp.sqrt(gm / (a*a*a))
    ecosE0, esinE0 = 1. - r0 / a, u / (n*a*a)

    return n, ecosE0, esinE0, a/r0


def find_transit_times_kepler(xast, vast, kast, dt, nitr):
    """find transit times via interpolation

        Note:
            This function is adapted from TTVFast https://github.com/kdeck/TTVFast, original scheme developed by Nesvorny et al. (2013, ApJ 777,3)

        Args:
            xast: astrocentric positions (Norbit, xyz)
            vast: astrocentric velocities (Norbit, xyz)
            kast: astrocentric GM
            dt: integration time step

        Returns:
            time to the transit center

    """
    n, ecosE0, esinE0, a_r0 = get_elements(xast, vast, kast)
    rsquared = jnp.sum(xast[:, :2]*xast[:, :2], axis=1)
    vsquared = jnp.sum(vast[:, :2]*vast[:, :2], axis=1)
    xdotv = jnp.sum(xast[:, :2]*vast[:, :2], axis=1)

    def dEstep_transit(dE, i):
        x2 = dE / 2.0
        sx2, cx2 = jnp.sin(x2), jnp.cos(x2)
        f = 1.0 - a_r0*2.0*sx2*sx2
        sx, cx = 2.0*sx2*cx2, cx2*cx2 - sx2*sx2
        g = (2.0*sx2*(esinE0*sx2 + cx2/a_r0))/n
        fp = 1.0 - cx*ecosE0 + sx*esinE0
        fdot = -(a_r0/fp)*n*sx
        fp2 = sx*ecosE0 + cx*esinE0
        gdot = 1.0-2.0*sx2*sx2/fp

        dgdotdz = -sx/fp+2.0*sx2*sx2/fp/fp*fp2
        dfdz = -a_r0*sx
        dgdz = 1.0/n*(sx*esinE0-(ecosE0-1.0)*cx)
        dfdotdz = -n*a_r0/fp*(cx+sx/fp*fp2)

        dotproduct = f*fdot*(rsquared)+g*gdot*(vsquared) + \
            (f*gdot+g*fdot)*(xdotv)
        dotproductderiv = dfdz*(gdot*xdotv+fdot*rsquared)+dfdotdz*(
            f*rsquared+g*xdotv)+dgdz*(fdot*xdotv+gdot*vsquared)+dgdotdz*(g*vsquared+f*xdotv)

        return dE - dotproduct/dotproductderiv, None

    dE0 = n * dt / 2.0
    dE, _ = scan(dEstep_transit, dE0, jnp.arange(nitr))
    x2 = dE / 2.0
    sx2, cx2 = jnp.sin(x2), jnp.cos(x2)
    sx = 2.0 * sx2 * cx2
    transitM = dE + esinE0 * 2.0 * sx2 * sx2 - sx * ecosE0

    return transitM / n


# map along the transit axis
find_transit_times_kepler_map = vmap(
    find_transit_times_kepler, (0, 0, 0, None, None), 0)


def find_transit_times_kepler_all(pidxarr, tcobsarr, t, xvjac, masses, nitr=3):
    """find transit times for all planets via interpolation

        Note: 
            Bug: this function sometimes fails for large dt for reason yet to be understood.

        Args:
            pidxarr: array of orbit index starting from 0 (Ntransit,)
            tcobsarray: flattened array of observed transit times (Ntransit,)
            t: times (Nstep,)
            xvjac: Jacobi positions and velocities (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,)
            nitr: number of Newton-Raphson iterations

        Returns:
            transit times (1D flattened array)

    """
    xjac, vjac = xvjac[:, 0, :, :], xvjac[:, 1, :, :]
    tcflag = get_tcflag(xjac, vjac)
    tcidx = find_tc_idx_map(t, tcflag, pidxarr, tcobsarr).ravel()

    tc_ahead, tc_behind = t[1:][tcidx], t[1:][tcidx-1]
    xvjac_ahead, xvjac_behind = xvjac[1:][tcidx], xvjac[1:][tcidx-1]

    # bring back the system by dt/2 so that the systems are at conclusions of the symplectic step
    # if the transit is not bracketed after this shift, advance the system by dt again
    dt = jnp.diff(t)[0]
    dt2 = 0.5 * dt
    xjac_ahead_mindt2, vjac_ahead_mindt2 = kepler_step_map(
        xvjac_ahead[:, 0, :, :], xvjac_ahead[:, 1, :, :], masses, -dt2)
    xjac_behind_mindt2, vjac_behind_mindt2 = kepler_step_map(
        xvjac_behind[:, 0, :, :], xvjac_behind[:, 1, :, :], masses, -dt2)
    xjac_ahead_plusdt2, vjac_ahead_plusdt2 = kick_kepler_map(
        xvjac_ahead[:, 0, :, :], xvjac_ahead[:, 1, :, :], masses, dt2)
    tcflag_mindt2 = get_g_map(xjac_ahead_mindt2, vjac_ahead_mindt2,
                              pidxarr) > 0.  # True if still bracketing the transit

    def func(x, y, z): return jnp.where(x, y, z)
    func_map = vmap(func, (0, 0, 0), 0)
    xjac_ahead = func_map(tcflag_mindt2, xjac_ahead_mindt2, xjac_ahead_plusdt2)
    xjac_behind = func_map(
        tcflag_mindt2, xjac_behind_mindt2, xjac_ahead_mindt2)
    vjac_ahead = func_map(tcflag_mindt2, vjac_ahead_mindt2, vjac_ahead_plusdt2)
    vjac_behind = func_map(
        tcflag_mindt2, vjac_behind_mindt2, vjac_ahead_mindt2)
    tc_ahead = jnp.where(tcflag_mindt2, tc_ahead - dt2, tc_ahead + dt2)
    tc_behind = jnp.where(tcflag_mindt2, tc_behind - dt2, tc_behind + dt2)

    xast_ahead, vast_ahead = jacobi_to_astrocentric(
        xjac_ahead, vjac_ahead, masses)
    xast_behind, vast_behind = jacobi_to_astrocentric(
        xjac_behind, vjac_behind, masses)

    kast = G * (masses[1:] + masses[0])
    kastarr = kast[pidxarr]

    tau_ahead = tc_ahead + jnp.diag(find_transit_times_kepler_map(
        xast_ahead, vast_ahead, kastarr, -dt, nitr)[:, pidxarr])
    tau_behind = tc_behind + jnp.diag(find_transit_times_kepler_map(
        xast_behind, vast_behind, kastarr, dt, nitr)[:, pidxarr])

    tc = ((tau_behind - tc_behind) * tau_ahead + (tc_ahead - tau_ahead)
          * tau_behind) / (dt + tau_behind - tau_ahead)

    return tc


""" Newton-Raphson method w/ for loop """
'''
def find_transit_times(t, x, v, a, j, tcobs, masses, nitr=5):
    """ find transit times (jit version)
    This requires tcobs, since the routine finds only transit times nearest to the observed ones.

        Args:
            t: times (Nstep,)
            x: positions in CoM frame (Nstep, Norbit, xyz)
            v: velocities in CoM frame (Nstep, Norbit, xyz)
            a: accelerations in CoM frame (Nstep, Norbit, xyz)
            j: index of the orbit (planet) for each transit times are computed
            tcobs: observed transit times for jth orbit (planet)
            masses: masses of the bodies (Nbody,), solar unit
            niter: number of Newton-Raphson iterations

        Returns:
            transit times for the jth orbit (planet)
            nearest to the observed ones

    """
    xastj, vastj, aastj = cm_to_astrocentric(x, v, a, j)
    gj, dotgj = get_gderivs(xastj, vastj, aastj)

    # get t, x, v where tcidx=True; difficult to make this compatible with jit
    # should be improved
    tcidx = (gj[1:] * gj[:-1] < 0) & (xastj[1:,2] > 0) & (dotgj[1:] > 0)
    _tc = jnp.where(tcidx, t[1:], -jnp.inf)
    idxsort = jnp.argsort(_tc)
    _tcsort = _tc[idxsort]
    tcidx1 = jnp.searchsorted(_tcsort, tcobs)
    tcidx2 = tcidx1 - 1
    tc1, tc2 = _tcsort[tcidx1], _tcsort[tcidx2]
    tcidx = jnp.where(jnp.abs(tcobs-tc1) < jnp.abs(tcobs-tc2), tcidx1, tcidx2)
    tc = _tcsort[tcidx]

    nrstep = - (gj / dotgj)[1:][idxsort][tcidx]
    xtc = x[1:,:,:][idxsort][tcidx]
    vtc = v[1:,:,:][idxsort][tcidx]

    def tcstep(xvs, i):
        xin, vin, step = xvs
        xtc, vtc, atc = hermite4_step_map(xin, vin, masses, step)
        xtc = jnp.transpose(xtc, axes=[2,0,1])
        vtc = jnp.transpose(vtc, axes=[2,0,1])
        atc = jnp.transpose(atc, axes=[2,0,1])
        _xastj, _vastj, _aastj = cm_to_astrocentric(xtc, vtc, atc, j)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        step = - _gj / _dotgj
        return [xtc, vtc, step], step

    _, steps = scan(tcstep, [xtc, vtc, nrstep], jnp.arange(nitr))
    tc += nrstep + jnp.sum(steps, axis=0)

    return tc


def find_transit_times_planets(t, x, v, a, tcobs, masses, nitr=5):
    """ find transit times: loop over each planet (should be modified)

        Args:
            t: times
            x: positions in CoM frame (Nstep, Norbit, xyz)
            v: velocities in CoM frame (Nstep, Norbit, xyz)
            a: accelerations in CoM frame (Nstep, Norbit, xyz)
            tcobs: list of observed transit times
            masses: masses of the bodies (in units of solar mass)

        Returns:
            model transit times (1D flattened array)

    """
    tcarr = jnp.array([])
    for j in range(len(masses)-1):
        tc = find_transit_times(t, x, v, a, j+1, tcobs[j], masses, nitr=nitr)
        tcarr = jnp.hstack([tcarr, tc])
    return tcarr
'''
