""" 4th-order Hermite integrator based on Kokubo, E., & Makino, J. 2004, PASJ, 56, 861
used for transit time computation
"""

#%%
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import scan
from .utils import *
from jax.config import config
config.update('jax_enable_x64', True)

#%%
#@jit
def get_derivs(x, v, masses):
    xjk = jnp.transpose(x[:,None] - x[None, :], axes=[0,2,1])
    vjk = jnp.transpose(v[:,None] - v[None, :], axes=[0,2,1])
    x2jk = jnp.sum(xjk * xjk, axis=1)[:,None,:]
    xvjk = jnp.sum(xjk * vjk, axis=1)[:,None,:]

    x2jk = jnp.where(x2jk!=0., x2jk, jnp.inf)
    x2jkinv = 1. / x2jk
    x2jkinv1p5 = x2jkinv * jnp.sqrt(x2jkinv)
    Xjk = - xjk * x2jkinv1p5
    dXjk = (- vjk + 3 * xvjk * xjk * x2jkinv) * x2jkinv1p5

    a = BIG_G * jnp.dot(Xjk, masses)
    adot = BIG_G * jnp.dot(dXjk, masses)

    return a, adot

#@jit
def predict(x, v, a, dota, dt):
    xp = x + dt * (v + 0.5 * dt * (a + dt * dota / 3.))
    vp = v + dt * (a + 0.5 * dt * dota)
    return xp, vp

#@jit
def correct(xp, vp, a1, dota1, a, dota, dt, alpha=7./6.):
    a02 = (-6 * (a - a1) - 2 * dt * (2 * dota + dota1)) / dt**2
    a03 = (12 * (a - a1) + 6 * dt * (dota + dota1)) / dt**3
    xc = xp + (dt**4 / 24.) * (a02 + alpha * a03 * dt / 5.)
    vc = vp + (dt**3 / 6.) * (a02 + a03 * dt / 4.)
    return xc, vc

#@jit
def hermite4_step(x, v, masses, dt):
    a, dota = get_derivs(x, v, masses)
    xp, vp = predict(x, v, a, dota, dt)
    a1, dota1 = get_derivs(xp, vp, masses)
    xc, vc = correct(xp, vp, a1, dota1, a, dota, dt)
    return xc, vc, a1

hermite4_step_vmap = jit(vmap(hermite4_step, (0,0,None,0), 2)) # xva, body idx, xyz, transit idx

#@jit
def integrate_xv(x, v, masses, times):
    dtarr = jnp.diff(times)

    def step(xvin, dt):
        xin, vin = xvin
        xout, vout, a1 = hermite4_step(xin, vin, masses, dt)
        return [xout, vout], jnp.array([xout, vout, a1])

    _, xv = scan(step, [x, v], dtarr)

    return times[1:], xv

#@jit
def integrate_elements(elements, masses, times, t_epoch):
    xrel_j, vrel_j = initialize_from_elements(elements, masses, t_epoch)
    xrel_ast, vrel_ast = jacobi_to_astrocentric(xrel_j, vrel_j, masses)
    x, v = astrocentric_to_cm(xrel_ast, vrel_ast, masses)
    t, xva = integrate_xv(x, v, masses, times)
    return t, xva

#%% transtit finding
#@jit
def get_gderivs(xastj, vastj, aastj):
    gj = jnp.sum(xastj[:,:2] * vastj[:,:2], axis=1)
    dotgj = jnp.sum(vastj[:,:2] * vastj[:,:2], axis=1) + jnp.sum(xastj[:,:2] * aastj[:,:2], axis=1)
    return gj, dotgj

def find_transit_times_nodata(t, x, v, a, j, masses, nitr=5):
    xastj, vastj, aastj = cm_to_astrocentric(x, v, a, j)
    gj, dotgj = get_gderivs(xastj, vastj, aastj)

    tcidx = (gj[1:] * gj[:-1] < 0) & (xastj[1:,2] > 0) & (dotgj[1:] > 0) # step after the sign was changed
    tc = t[1:][tcidx]

    nrstep = - (gj / dotgj)[1:][tcidx]
    xtc = x[1:,:,:][tcidx]
    vtc = v[1:,:,:][tcidx]

    for i in range(nitr):
        tc += nrstep
        xtc, vtc, atc = hermite4_step_vmap(xtc, vtc, masses, nrstep)
        xtc = jnp.transpose(xtc, axes=[2,0,1])
        vtc = jnp.transpose(vtc, axes=[2,0,1])
        atc = jnp.transpose(atc, axes=[2,0,1])
        _xastj, _vastj, _aastj = cm_to_astrocentric(xtc, vtc, atc, j)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        nrstep = - _gj / _dotgj

    return tc

#@jit
def findidx(arr, x):
    return jnp.argmin(jnp.abs(arr - x))
findidx_map = jit(vmap(findidx, (None,0), 0))

def find_transit_times(t, x, v, a, j, tcobs, masses, nitr=5):
    """ routine to find transit times

        Args:
            t: times
            x, v, a: position, velocity, acceleration (Ntime, Norbit, Nxyz)
            j: orbit index
            tcobs: observed transit times for jth orbit (planet)
            masses: masses of the bodies

        Returns:
            N-body transit times nearest to the observed ones

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
        xtc, vtc, atc = hermite4_step_vmap(xin, vin, masses, step)
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

#@jit
def find_transit_times_planets(t, xva, tcobs, masses):
    """ find transit times for every planet (should make this part mappable to eliminate for loop?)

        Args:
            t: times
            xva: position, velocity, acceleration (Nstep, xva, Norbit, xyz)
            tcobs: observed transit times, list(!) of length Norbit
            masses: masses of the bodies

        Returns:
            N-body transit times (1D array)

    """
    tcarr = jnp.array([])
    for j in range(1,len(masses)):
        #tc = find_transit_times(t, xva, j, tcobs[j-1], masses)
        tc = find_transit_times(t, xva[:,0,:,:], xva[:,1,:,:], xva[:,2,:,:], j, tcobs[j-1], masses)
        tcarr = jnp.hstack([tcarr, tc])
    return tcarr
