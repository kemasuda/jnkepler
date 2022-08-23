""" routines for finding transit times
"""
__all__ = [
    "find_transit_times_nodata", "find_transit_times", "find_transit_times_planets",
    "find_transit_times_all"
]

#%%
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import scan
from .conversion import cm_to_astrocentric
from .hermite4 import hermite4_step_map
from .utils import findidx_map
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


def find_transit_times_nodata(t, x, v, a, j, masses, nitr=5):
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
            nearest to  the observed ones

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

""" corresponds to integrate_elemetns in hermite4
def find_transit_times_planets(t, xva, tcobs, masses):
        find transit times for every planet (should make this part mappable to eliminate for loop?)

        Args:
            t: times
            xva: position, velocity, acceleration (Nstep, xva, Norbit, xyz)
            tcobs: observed transit times, list(!) of length Norbit
            masses: masses of the bodies

        Returns:
            N-body transit times (1D array)


    tcarr = jnp.array([])
    for j in range(1,len(masses)):
        tc = find_transit_times(t, xva[:,0,:,:], xva[:,1,:,:], xva[:,2,:,:], j, tcobs[j-1], masses)
        tcarr = jnp.hstack([tcarr, tc])
    return tcarr
"""


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

"""
def get_ttvs(self, elements, masses):
    compute model transit times given orbtal elements and masses

        Args:
            self: JaxTTV class
            elements: orbital elements in JaxTTV format
            masses: masses of the bodies (in units of solar mass)

        Returns:
            model transit times (1D flattened array)
            fractional energy change

    x0, v0 = initialize_from_elements(elements, masses, self.t_start)
    t, xv = integrate_xv(x0, v0, masses, self.times)
    x, v, a = xvjac_to_xvacm(xv, masses)
    etot = get_energy_map(x, v, masses)
    tpars = find_transit_times_planets(t, x, v, a, self.tcobs, masses)
    return tpars, etot[-1]/etot[0]-1.
"""

## New algorithm
def get_tcflag(t, x, v, a):
    """ get boolean flags for transit centers

        Args:
            t: times (Nstep)
            x: positions in CM frame (Nstep, Norbit, xyz)
            v: velocities in CM frame (Nstep, Norbit, xyz)
            a: accels in CM frame (Nstep, Norbit, xyz)

        Returns:
            tcflag (Nstep,Norbit): True when the orbit has just crossed the transit center
            -(g /dot g), (Nstep,Norbit): Newton-Raphson step

    """
    xast = x[:,1:,:] - x[:,0,:][:,None,:] # Nstep, Norbit, xyz
    vast = v[:,1:,:] - v[:,0,:][:,None,:]
    aast = a[:,1:,:] - a[:,0,:][:,None,:]
    g = jnp.sum(xast[:,:,:2] * vast[:,:,:2], axis=2) # Nstep, Norbit
    dotg = jnp.sum(vast[:,:,:2] * vast[:,:,:2], axis=2) + jnp.sum(xast[:,:,:2] * aast[:,:,:2], axis=2)
    tcflag = (g[1:] * g[:-1] < 0) & (xast[1:,:,2] > 0) & (dotg[1:] > 0)
    return tcflag, -(g / dotg)

def find_tc_init_idx(t, tcflag, nrstep, j, tcobs):
    """ find indices and NR steps corresponding transit centers

        Args:
            t: times (Nstep,)
            tcflag: True when the orbit has crossed the transit center (Nstep,Norbit)
            nrstep: NR step (Nstep,Norbit)
            j: orbit (planet) index starting from 0
            tcobs: observed transit times for jth orbit

        Returns:
            tcidx: indices of the transit centers nearest to tcobs, (Ntransit,)
            NR step

    """
    tc_candidates = jnp.where(tcflag[:,j], t[1:], -jnp.inf)
    tcidx = findidx_map(tc_candidates, jnp.atleast_1d(tcobs))
    return tcidx, nrstep[1:][tcidx][:,j]

# map along the transit axis
find_tc_init_idx_map = jit(vmap(find_tc_init_idx, (None,None,None,0,0), 0))


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
get_nrstep_map = jit(vmap(get_nrstep, (0,0,0,0), 0))

#@partial(jit, static_argnums=(0,))
#def find_transit_times_all(self, t, x, v, a, masses, nitr=5):
#self.pidx.astype(int)-1
def find_transit_times_all(pidxarr, tcobsarr, t, x, v, a, masses, nitr=5):
    print ("new")
    tcflag, nrstep = get_tcflag(t, x, v, a)
    tcidx, nrstep_init = find_tc_init_idx_map(t, tcflag, nrstep, pidxarr, tcobsarr)
    tcidx, nrstep_init = tcidx.ravel(), nrstep_init.ravel()

    tc = t[1:][tcidx]
    x_init = x[1:,:,:][tcidx]
    v_init = v[1:,:,:][tcidx]

    def tcstep(xvs, i):
        xin, vin, step = xvs
        xtc, vtc, atc = hermite4_step_map(xin, vin, masses, step)
        xtc = jnp.transpose(xtc, axes=[2,0,1])
        vtc = jnp.transpose(vtc, axes=[2,0,1])
        atc = jnp.transpose(atc, axes=[2,0,1])
        step = get_nrstep_map(xtc, vtc, atc, pidxarr)
        return [xtc, vtc, step], step

    _, steps = scan(tcstep, [x_init, v_init, nrstep_init], jnp.arange(nitr))
    tc += nrstep_init + jnp.sum(steps, axis=0)

    return tc
