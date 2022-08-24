""" routines for finding transit times
"""
__all__ = [
    "find_transit_times_single", "find_transit_times_all"
]

#%%
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import scan
from functools import partial
from .conversion import cm_to_astrocentric, xvjac_to_xvacm
from .hermite4 import hermite4_step_map
from .utils import findidx_map, get_energy_map
from jax.config import config
config.update('jax_enable_x64', True)


def get_gderivs(xastj, vastj, aastj):
    """ time derivatives of the 'g' function (squared sky-projected star-planet distance)
    Here g = x*x + y*y

        Args:
            xastj: astrocentric positions (Norbit, xyz)
            vastj: astrocentric velocities (Norbit, xyz)
            aastj: astrocentric accelerations (Norbit, xyz)

        Returns:
            values of g, dg/dt (Norbit,)

    """
    gj = jnp.sum(xastj[:,:2] * vastj[:,:2], axis=1)
    dotgj = jnp.sum(vastj[:,:2] * vastj[:,:2], axis=1) + jnp.sum(xastj[:,:2] * aastj[:,:2], axis=1)
    return gj, dotgj


def find_transit_times_single(t, x, v, a, j, masses, nitr=5):
    """ find transit times (cannot be jitted)

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

    tcidx = (gj[1:] * gj[:-1] < 0) & (xastj[1:,2] > 0) & (dotgj[1:] > 0) # step after the sign was changed
    tc = t[1:][tcidx]

    nrstep = - (gj / dotgj)[1:][tcidx]
    xtc = x[1:,:,:][tcidx]
    vtc = v[1:,:,:][tcidx]

    for i in range(nitr):
        tc += nrstep
        xtc, vtc, atc = hermite4_step_map(xtc, vtc, masses, nrstep)
        xtc = jnp.transpose(xtc, axes=[2,0,1])
        vtc = jnp.transpose(vtc, axes=[2,0,1])
        atc = jnp.transpose(atc, axes=[2,0,1])
        _xastj, _vastj, _aastj = cm_to_astrocentric(xtc, vtc, atc, j)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        nrstep = - _gj / _dotgj

    return tc


""" new algorithm w/o for loop """
def get_tcflag(t, xjac, vjac):
    """ find times just after the transit centers using *Jacobi* coordinates

        Args:
            t: times (Nstep,)
            xjac: jacobi positions (Nstep,Norbit,xyz)
            vjac: jacobi velocities (Nstep,Norbit,xyz)

        Returns:
            tcflag: True if the time is just after the transit center (Nstep-1,)

    """
    g = jnp.sum(xjac[:,:,:2] * vjac[:,:,:2], axis=2) # Nstep, Norbit
    tcflag = (g[:-1] < 0) & (g[1:] > 0) & (xjac[1:,:,2] > 0)
    return tcflag


def find_tc_idx(t, tcflag, j, tcobs):
    """ find indices for times where tcflag is True

        Args:
            t: times (Nstep,)
            tcflag: True if the time is just after the transit center (Nstep-1,)
            j: orbit (planet) index
            tcobs: transit times for jth orbit (planet)

        Returns:
            indices of times cloeset to transit centers (Nstep-1,)
            should be put into times[1:], x[1:], etc.

    """
    tc_candidates = jnp.where(tcflag[:,j], t[1:], -jnp.inf)
    tcidx = findidx_map(tc_candidates, jnp.atleast_1d(tcobs))
    return tcidx

# map along the transit axis
find_tc_idx_map = vmap(find_tc_idx, (None,None,0,0), 0)


def get_nrstep(x, v, a, j):
    """ compute NR step for jth orbit (planet)

        Args:
            x: positions in CM frame (Norbit, xyz)
            v: velocities in CM frame (Norbit, xyz)
            a: accels in CM frame (Norbit, xyz)
            j: orbit (planet) index, starting from 0

        Returns:
            NR step for jth orbit (planet)

    """
    xastj = x[j+1,:] - x[0,:]
    vastj = v[j+1,:] - v[0,:]
    aastj = a[j+1,:] - a[0,:]
    gj = jnp.sum(xastj[:2] * vastj[:2])
    dotgj = jnp.sum(vastj[:2] * vastj[:2]) + jnp.sum(xastj[:2] * aastj[:2])
    stepj = - gj / dotgj
    return stepj

# map along the transit axis
get_nrstep_map = vmap(get_nrstep, (0,0,0,0), 0)


def find_transit_times_all(pidxarr, tcobsarr, t, xvjac, masses, nitr=5):
    """ find transit times for all planets

        Args:
            pidxarr: array of orbit index starting from 0 (Ntransit,)
            tcobsarray: flattened array of observed transit times (Ntransit,)
            t: times (Nstep,)
            xvjac: Jacobi positions and velocities (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,)
            nitr: number of Newton-Raphson iterations

        Returns:
            transit times (1D flattened array)
            total energies around transit times

    """
    xjac, vjac = xvjac[:,0,:,:], xvjac[:,1,:,:]
    tcflag = get_tcflag(t, xjac, vjac)
    tcidx = find_tc_idx_map(t, tcflag, pidxarr, tcobsarr).ravel()

    tc = t[1:][tcidx]
    xvjac_init = xvjac[1:][tcidx]
    xcm_init, vcm_init, acm_init = xvjac_to_xvacm(xvjac_init, masses)
    nrstep_init = get_nrstep_map(xcm_init, vcm_init, acm_init, pidxarr)

    def tcstep(xvs, i):
        xin, vin, step = xvs
        xtc, vtc, atc = hermite4_step_map(xin, vin, masses, step)
        xtc = jnp.transpose(xtc, axes=[2,0,1])
        vtc = jnp.transpose(vtc, axes=[2,0,1])
        atc = jnp.transpose(atc, axes=[2,0,1])
        step = get_nrstep_map(xtc, vtc, atc, pidxarr)
        return [xtc, vtc, step], step

    _, steps = scan(tcstep, [xcm_init, vcm_init, nrstep_init], jnp.arange(nitr))
    tc += nrstep_init + jnp.sum(steps, axis=0)

    etot = get_energy_map(xcm_init, vcm_init, masses) # total energies around transit times

    return tc, etot


""" algorithm w/ for loop """
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
