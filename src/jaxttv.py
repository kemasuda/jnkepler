#%%
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from functools import partial
import matplotlib.pyplot as plt

#%%
from jax.config import config
config.update('jax_enable_x64', True)

#%%
BIG_G = 2.959122082855911e-4
sidx = 0
xkey, vkey, akey = 0, 1, 2

#%%
@jit
def tic_to_u(tic, period, ecc, omega):
    tanw2 = jnp.tan(0.5 * omega)
    u = 2 * jnp.arctan( jnp.sqrt((1.-ecc)/(1.+ecc)) * (1.-tanw2)/(1.+tanw2) )
    return u


@jit
def xvrel_from_elements(porb, ecc, inc, omega, lnode, u, mass):
    cosu, sinu = jnp.cos(u), jnp.sin(u)
    cosw, sinw, cosO, sinO, cosi, sini = jnp.cos(omega), jnp.sin(omega), jnp.cos(lnode), jnp.sin(lnode), jnp.cos(inc), jnp.sin(inc)

    n = 2 * jnp.pi / porb
    na = (n * BIG_G * mass) ** (1./3.)
    R = 1.0 - ecc * cosu

    Pvec = jnp.array([cosw*cosO - sinw*sinO*cosi, cosw*sinO + sinw*cosO*cosi, sinw*sini])
    Qvec = jnp.array([-sinw*cosO - cosw*sinO*cosi, -sinw*sinO + cosw*cosO*cosi, cosw*sini])
    x, y = cosu - ecc, jnp.sqrt(1.-ecc*ecc) * sinu
    vx, vy = -sinu, jnp.sqrt(1.-ecc*ecc) * cosu

    xrel = (na / n) * (Pvec * x + Qvec * y)
    vrel = (na / R) * (Pvec * vx + Qvec * vy)

    return xrel, vrel

#%%
@jit
def initialize_from_elements(elements, masses):
    xrel, vrel = [], []
    for j in range(len(elements)):
        porb, ecc, inc, omega, lnode, tic = elements[j]
        u = tic_to_u(tic, porb, ecc, omega)
        xr, vr = xvrel_from_elements(porb, ecc, inc, omega, lnode, u, jnp.sum(masses[:j+2]))
        xrel.append(xr)
        vrel.append(vr)
    xrel, vrel = jnp.array(xrel), jnp.array(vrel)
    return xrel, vrel

#%%
@jit
def jacobi_to_astrocentric(xrel_j, vrel_j, masses):
    nbody = len(masses)
    mmat = jnp.eye(nbody-1) + jnp.tril(jnp.array([jnp.ones(nbody-1)*masses[j-1]/jnp.sum(masses[:j]) for j in range(2, nbody+1)]).T, k=-1)
    return mmat@xrel_j, mmat@vrel_j

#%% move to CoM frame
@jit
def astrocentric_to_cm(xrel_ast, vrel_ast, masses):
    mtot = jnp.sum(masses)
    xcm_ast = jnp.sum(masses[1:][:,None] * xrel_ast, axis=0) / mtot
    vcm_ast = jnp.sum(masses[1:][:,None] * vrel_ast, axis=0) / mtot
    xcm = jnp.vstack([-xcm_ast, xrel_ast - xcm_ast])
    vcm = jnp.vstack([-vcm_ast, vrel_ast - vcm_ast])
    return xcm, vcm

#%%
@jit
def get_derivs(x, v, masses):
    xjk = jnp.transpose(x[:,None] - x[None, :], axes=[0,2,1])
    vjk = jnp.transpose(v[:,None] - v[None, :], axes=[0,2,1])
    x2jk = jnp.sum(xjk * xjk, axis=1)[:,None,:]
    xvjk = jnp.sum(xjk * vjk, axis=1)[:,None,:]
    Xjk = - jnp.where(xjk!=0., xjk / x2jk**1.5, 0)
    dXjk = - jnp.where(xjk!=0., vjk / x2jk**1.5, 0) + jnp.where(xjk!=0., 3 * xvjk * xjk / x2jk**2.5, 0)
    a = BIG_G * jnp.dot(Xjk, masses)
    adot = BIG_G * jnp.dot(dXjk, masses)
    return a, adot

@jit
def predict(x, v, a, dota, dt):
    xp = x + dt * (v + 0.5 * dt * (a + dt * dota / 3.))
    vp = v + dt * (a + 0.5 * dt * dota)
    return xp, vp

@jit
def correct(xp, vp, a, dota, dt, alpha=7./6.):
    a1, dota1 = get_derivs(xp, vp, masses)
    a02 = (-6 * (a - a1) - 2 * dt * (2 * dota + dota1)) / dt**2
    a03 = (12 * (a - a1) + 6 * dt * (dota + dota1)) / dt**3
    xc = xp + (dt**4 / 24.) * (a02 + alpha * a03 * dt / 5.)
    vc = vp + (dt**3 / 6.) * (a02 + a03 * dt / 4.)
    return xc, vc, a1

@jit
def hermite4_step(x, v, masses, dt):
    a, dota = get_derivs(x, v, masses)
    xp, vp = predict(x, v, a, dota, dt)
    xc, vc, a1 = correct(xp, vp, a, dota, dt)
    return xc, vc, a1

#%%
@jit
def integrate(x, v, masses, times):
    dtarr = jnp.diff(times)

    def step(xvin, dt):
        xin, vin = xvin
        xout, vout, a1 = hermite4_step(xin, vin, masses, dt)
        return [xout, vout], jnp.array([xout, vout, a1])

    _, xv = scan(step, [x, v], dtarr)

    return times[1:], xv

#%%
@jit
def get_gderivs(xastj, vastj, aastj):
    gj = jnp.sum(xastj[:,:2] * vastj[:,:2], axis=1)
    dotgj = jnp.sum(vastj[:,:2] * vastj[:,:2], axis=1) + jnp.sum(xastj[:,:2] * aastj[:,:2], axis=1)
    return gj, dotgj

hermite4_step_vmap = jit(vmap(hermite4_step, (0,0,None,0), 2)) # xva, body idx, xyz, transit idx

def find_transit_times_nodata(t, xva, j, nitr=3):
    def cm_to_astrocentric(x, v, a):
        xastj = x[:,j,:] - x[:,sidx,:]
        vastj = v[:,j,:] - v[:,sidx,:]
        aastj = a[:,j,:] - a[:,sidx,:]
        return xastj, vastj, aastj

    x, v, a = xva[:,xkey,:,:], xva[:,vkey,:,:], xva[:,akey,:,:]
    xastj, vastj, aastj = cm_to_astrocentric(x, v, a)
    gj, dotgj = get_gderivs(xastj, vastj, aastj)

    tcidx = (gj[1:] * gj[:-1] < 0) & (xastj[1:,2] > 0) & (dotgj[1:] > 0)# step after the sign was changed
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
        _xastj, _vastj, _aastj = cm_to_astrocentric(xtc, vtc, atc)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        nrstep = - _gj / _dotgj

    return tc, gj

@jit
def find_transit_times(t, xva, j, tcobs, nitr=3):
    def cm_to_astrocentric(x, v, a):
        xastj = x[:,j,:] - x[:,sidx,:]
        vastj = v[:,j,:] - v[:,sidx,:]
        aastj = a[:,j,:] - a[:,sidx,:]
        return xastj, vastj, aastj

    x, v, a = xva[:,xkey,:,:], xva[:,vkey,:,:], xva[:,akey,:,:]
    xastj, vastj, aastj = cm_to_astrocentric(x, v, a)
    gj, dotgj = get_gderivs(xastj, vastj, aastj)

    #tcidx = (gj[1:] * gj[:-1] < 0) & (xastj[1:,2] > 0) & (dotgj[1:] > 0)# step after the sign was changed
    tcidx = jnp.searchsorted(t, tcobs)
    tc = t[1:][tcidx]

    nrstep = - (gj / dotgj)[1:][tcidx]
    xtc = x[1:,:,:][tcidx]
    vtc = v[1:,:,:][tcidx]

    """
    def tcstep(xvs, i):
        xin, vin, step = xvs
        xtc, vtc, atc = hermite4_step_vmap(xin, vin, masses, step)
        xtc = jnp.transpose(xtc, axes=[2,0,1])
        vtc = jnp.transpose(vtc, axes=[2,0,1])
        atc = jnp.transpose(atc, axes=[2,0,1])
        _xastj, _vastj, _aastj = cm_to_astrocentric(xtc, vtc, atc)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        step = - _gj / _dotgj
        return [xtc, vtc, step], step

    _, steps = scan(tcstep, [xtc, vtc, nrstep], jnp.arange(nitr))
    tc += nrstep + jnp.sum(steps, axis=0)
    """

    #"""
    for i in range(nitr):
        tc += nrstep
        xtc, vtc, atc = hermite4_step_vmap(xtc, vtc, masses, nrstep)
        xtc = jnp.transpose(xtc, axes=[2,0,1])
        vtc = jnp.transpose(vtc, axes=[2,0,1])
        atc = jnp.transpose(atc, axes=[2,0,1])
        _xastj, _vastj, _aastj = cm_to_astrocentric(xtc, vtc, atc)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        nrstep = - _gj / _dotgj
    #"""

    return tc, gj

#%%
class jaxttv:
    def __init__(self, t_start, t_end, dt):
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.times = jnp.arange(t_start, t_end, dt)
        self.tcobs = []

    def set_tcobs(self, tcobs):
        self.tcobs = tcobs

    #@partial(jit, static_argnums=(0,))
    def get_ttvs_nodata(self, elements, masses):
        xrel_j, vrel_j = initialize_from_elements(elements, masses)
        xrel_ast, vrel_ast = jacobi_to_astrocentric(xrel_j, vrel_j, masses)
        x, v = astrocentric_to_cm(xrel_ast, vrel_ast, masses)
        t, xva = integrate(x, v, masses, self.times)

        tcarr = []
        for pidx in range(1, len(masses)):
            tc, gj = find_transit_times_nodata(t, xva, pidx)
            tcarr.append(tc)

        return tcarr

    @partial(jit, static_argnums=(0,))
    def get_ttvs(self, elements, masses):
        xrel_j, vrel_j = initialize_from_elements(elements, masses)
        xrel_ast, vrel_ast = jacobi_to_astrocentric(xrel_j, vrel_j, masses)
        x, v = astrocentric_to_cm(xrel_ast, vrel_ast, masses)
        t, xva = integrate(x, v, masses, self.times)

        tcarr = []
        for pidx in range(1, len(masses)):
            tc, gj = find_transit_times(t, xva, pidx, self.tcobs[pidx-1])
            tcarr.append(tc)

        return tcarr


#%%
dt = 0.01 * 10
t_start, t_end = 0, 1e4
jttv = jaxttv(t_start, t_end, dt)

#%%
elements = jnp.array([[365.25, 0.4, jnp.pi*0.5, 0.1*jnp.pi, 0.1*jnp.pi, 40], [365.25*2., 0., jnp.pi*0.5, 0.1*jnp.pi, 0.1*jnp.pi, 40]])#[:1]
masses = jnp.array([1, 3e-6, 300e-6])#[:2]

#%%
tcobs = jttv.get_ttvs_nodata(elements, masses)
jttv.set_tcobs(tcobs)

#%%
plt.plot(tcobs[0], '.')
plt.plot(tcobs[1], '.')

#%%
jttv.tcobs

#%%
%time tpars = jttv.get_ttvs(elements, masses)

#%%
%timeit tpars = jttv.get_ttvs(elements, masses)

#%%
tpars

#%%
jttv.tcobs
