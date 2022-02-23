#%%
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from functools import partial
from jkepler.kepler.markley import get_E
import matplotlib.pyplot as plt

#%%
from jax.config import config
config.update('jax_enable_x64', True)

#%%
BIG_G = 2.959122082855911e-4

#%%
sidx = 0
xkey, vkey, akey = 0, 1, 2
# elements format: porb, ecosw, esinw, cosi, lnode, tic
# masses format: stellar mass, planetary masses in units of M_sun
# xva format: x/v/a, time index, body index, x/y/z

#%%
@jit
def tic_to_u(tic, period, ecc, omega, t_epoch):
    tanw2 = jnp.tan(0.5 * omega)
    uic = 2 * jnp.arctan( jnp.sqrt((1.-ecc)/(1.+ecc)) * (1.-tanw2)/(1.+tanw2) ) # u at t=tic
    #tau = tic -  period / (2 * jnp.pi) * (uic - ecc*jnp.sin(uic))
    #M_epoch = - 2 * jnp.pi / period * tic + uic - ecc * jnp.sin(uic) # M at t=0
    M_epoch = 2 * jnp.pi / period * (t_epoch - tic) + uic - ecc * jnp.sin(uic) # M at t=0
    u_epoch = get_E(M_epoch, ecc)
    return u_epoch

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
def initialize_from_elements(elements, masses, t_epoch):
    xrel, vrel = [], []
    for j in range(len(elements)):
        #porb, ecc, inc, omega, lnode, tic = elements[j]

        porb, ecosw, esinw, cosi, lnode, tic = elements[j]
        ecc = jnp.sqrt(ecosw**2 + esinw**2)
        omega = jnp.arctan2(esinw, ecosw)
        inc = jnp.arccos(cosi)

        u = tic_to_u(tic, porb, ecc, omega, t_epoch)
        xr, vr = xvrel_from_elements(porb, ecc, inc, omega, lnode, u, jnp.sum(masses[:j+2]))
        xrel.append(xr)
        vrel.append(vr)
    xrel, vrel = jnp.array(xrel), jnp.array(vrel)
    return xrel, vrel

#%%
@jit
def jacobi_to_astrocentric(xrel_j, vrel_j, masses):
    nbody = len(masses)
    #mmat = jnp.eye(nbody-1) + jnp.tril(jnp.array([jnp.ones(nbody-1)*masses[j-1]/jnp.sum(masses[:j]) for j in range(2, nbody+1)]).T, k=-1)
    mmat = jnp.eye(nbody-1) + jnp.tril(jnp.tile(masses[1:] / jnp.cumsum(masses)[1:], (nbody-1,1)), k=-1)
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
    #Xjk = - jnp.where(xjk!=0., xjk / x2jk**1.5, 0)
    x2jk = jnp.where(x2jk!=0., x2jk, 1)
    x2jkinv = jnp.where(x2jk!=0., 1. / x2jk, 0) # none of x2jk should contain zero, even in the first case (otherwise jit grad fails); that's why this must have been taken care of in the above line
    Xjk = - xjk * x2jkinv**1.5
    dXjk = - vjk * x2jkinv**1.5 + 3 * xvjk * xjk * x2jkinv**2.5
    a = BIG_G * jnp.dot(Xjk, masses)
    adot = BIG_G * jnp.dot(dXjk, masses)
    return a, adot

@jit
def predict(x, v, a, dota, dt):
    xp = x + dt * (v + 0.5 * dt * (a + dt * dota / 3.))
    vp = v + dt * (a + 0.5 * dt * dota)
    return xp, vp

@jit
def correct(xp, vp, a1, dota1, a, dota, dt, alpha=7./6.):
    #a1, dota1 = get_derivs(xp, vp, masses)
    a02 = (-6 * (a - a1) - 2 * dt * (2 * dota + dota1)) / dt**2
    a03 = (12 * (a - a1) + 6 * dt * (dota + dota1)) / dt**3
    xc = xp + (dt**4 / 24.) * (a02 + alpha * a03 * dt / 5.)
    vc = vp + (dt**3 / 6.) * (a02 + a03 * dt / 4.)
    #return xc, vc, a1
    return xc, vc

@jit
def hermite4_step(x, v, masses, dt):
    a, dota = get_derivs(x, v, masses)
    xp, vp = predict(x, v, a, dota, dt)
    a1, dota1 = get_derivs(xp, vp, masses)
    xc, vc = correct(xp, vp, a1, dota1, a, dota, dt)
    return xc, vc, a1

#%%
@jit
def integrate_xv(x, v, masses, times):
    dtarr = jnp.diff(times)

    def step(xvin, dt):
        xin, vin = xvin
        xout, vout, a1 = hermite4_step(xin, vin, masses, dt)
        return [xout, vout], jnp.array([xout, vout, a1])

    _, xv = scan(step, [x, v], dtarr)

    return times[1:], xv

#%% xva: time, xva, object, xyz
@jit
def integrate_elements(elements, masses, times, t_epoch):
    xrel_j, vrel_j = initialize_from_elements(elements, masses, t_epoch)
    xrel_ast, vrel_ast = jacobi_to_astrocentric(xrel_j, vrel_j, masses)
    x, v = astrocentric_to_cm(xrel_ast, vrel_ast, masses)
    t, xva = integrate_xv(x, v, masses, times)
    return t, xva

#%%
@jit
def get_gderivs(xastj, vastj, aastj):
    gj = jnp.sum(xastj[:,:2] * vastj[:,:2], axis=1)
    dotgj = jnp.sum(vastj[:,:2] * vastj[:,:2], axis=1) + jnp.sum(xastj[:,:2] * aastj[:,:2], axis=1)
    return gj, dotgj

hermite4_step_vmap = jit(vmap(hermite4_step, (0,0,None,0), 2)) # xva, body idx, xyz, transit idx

def find_transit_times_nodata(t, xva, j, masses, nitr=3):
    def cm_to_astrocentric(x, v, a):
        xastj = x[:,j,:] - x[:,sidx,:]
        vastj = v[:,j,:] - v[:,sidx,:]
        aastj = a[:,j,:] - a[:,sidx,:]
        return xastj, vastj, aastj

    x, v, a = xva[:,xkey,:,:], xva[:,vkey,:,:], xva[:,akey,:,:]
    xastj, vastj, aastj = cm_to_astrocentric(x, v, a)
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
        _xastj, _vastj, _aastj = cm_to_astrocentric(xtc, vtc, atc)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        nrstep = - _gj / _dotgj

    return tc, gj

@jit
def find_transit_times(t, xva, j, tcobs, masses, nitr=3):
    def cm_to_astrocentric(x, v, a):
        xastj = x[:,j,:] - x[:,sidx,:]
        vastj = v[:,j,:] - v[:,sidx,:]
        aastj = a[:,j,:] - a[:,sidx,:]
        return xastj, vastj, aastj

    x, v, a = xva[:,xkey,:,:], xva[:,vkey,:,:], xva[:,akey,:,:]
    xastj, vastj, aastj = cm_to_astrocentric(x, v, a)
    gj, dotgj = get_gderivs(xastj, vastj, aastj)

    #tcidx = (gj[1:] * gj[:-1] < 0) & (xastj[1:,2] > 0) & (dotgj[1:] > 0) # step after the sign was changed; this kind of indexing doesn't work in jit
    tcidx = jnp.searchsorted(t, tcobs)
    tc = t[1:][tcidx]

    nrstep = - (gj / dotgj)[1:][tcidx]
    xtc = x[1:,:,:][tcidx]
    vtc = v[1:,:,:][tcidx]

    #"""
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
    #"""

    """
    for i in range(nitr):
        tc += nrstep
        xtc, vtc, atc = hermite4_step_vmap(xtc, vtc, masses, nrstep)
        xtc = jnp.transpose(xtc, axes=[2,0,1])
        vtc = jnp.transpose(vtc, axes=[2,0,1])
        atc = jnp.transpose(atc, axes=[2,0,1])
        _xastj, _vastj, _aastj = cm_to_astrocentric(xtc, vtc, atc)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        nrstep = - _gj / _dotgj
    """

    return tc, gj

#%% compute total energy of the system in CM frame unit: Msun*(AU/day)^2 */
@jit
def get_energy(x, v, masses):
    K = jnp.sum(0.5 * masses * jnp.sum(v*v, axis=1))
    """
    U = 0
    for i in range(jttv.nbody):
        for j in range(i):
            d = jnp.sqrt(jnp.sum((x[i,:] - x[j,:])**2))
            U -= BIG_G * masses[i] * masses[j] / d
    """
    X = x[:,None] - x[None,:]
    M = masses[:,None] * masses[None,:]
    U = -BIG_G * jnp.sum(M * jnp.tril(1./jnp.sqrt(jnp.sum(X*X, axis=2)), k=-1))
    return K + U

get_energy_vmap = jit(vmap(get_energy, (0,0,None), 0))
#emap = get_energy_vmap(xva[:,xkey,:,:], xva[:,vkey,:,:], masses)


#%%
class jaxttv:
    def __init__(self, t_start, t_end, dt):
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.times = jnp.arange(t_start, t_end, dt)

    def set_tcobs(self, tcobs, p_init, errorobs=None):
        self.tcobs = tcobs
        self.tcobs_flatten = np.hstack([t for t in tcobs])
        self.nplanet = len(tcobs)
        self.nbody = len(tcobs) + 1
        self.p_init = p_init
        if errorobs is None:
            self.errorobs_flatten = jnp.ones_like(self.tcobs_flatten)
        else:
            self.errorobs_flatten = np.hstack([e for e in errorobs])

        pidx, tcobs_linear, ttvamp = np.array([]), np.array([]), np.array([])
        for j in range(len(tcobs)):
            tc = tcobs[j]
            pidx = np.r_[pidx, np.ones_like(tc)*(j+1)]

            m = np.round((tc - tc[0]) / p_init[j])
            pfit, t0fit = np.polyfit(m, tc, deg=1)
            tc_linear = t0fit + m*pfit
            tcobs_linear = jnp.r_[tcobs_linear, tc_linear]

            ttv = tc - tc_linear
            ttvamp = jnp.r_[ttvamp, np.max(ttv)-np.min(ttv)]

        self.pidx = pidx
        self.tcobs_linear = tcobs_linear
        self.ttvamp = ttvamp

        print ("# integration starts at:".ljust(35) + "%.2f"%self.t_start)
        print ("# first transit time in data:".ljust(35) + "%.2f"%np.min(self.tcobs_flatten))
        print ("# last transit time in data:".ljust(35) + "%.2f"%np.max(self.tcobs_flatten))
        print ("# integration ends at:".ljust(35) + "%.2f"%self.t_end)
        print ("# integration time step:".ljust(35) + "%.4f (1/%d of innermost period)"%(self.dt, np.min(p_init)/self.dt))
        #print ("# first transit time: %f, last transit time: %f."%(np.min(tcobs_flatten), np.max(tcobs_flatten)))
        #print ("# integration from %f to %f with step %f."%(self.t_start, self.t_end, self.dt))

    def update_period(self):
        p_new = []
        for j in range(self.nplanet):
            tc = self.tcobs[j]
            m = np.round((tc - tc[0]) / self.p_init[j])
            pfit, t0fit = np.polyfit(m, tc, deg=1)
            p_new.append(pfit)
        return np.array(p_new)

    def get_ttvs_nodata(self, elements, masses, t_start, t_end, dt, return_xva=False):
        """
        times = jnp.arange(t_start, t_end, dt)
        xrel_j, vrel_j = initialize_from_elements(elements, masses)
        xrel_ast, vrel_ast = jacobi_to_astrocentric(xrel_j, vrel_j, masses)
        x, v = astrocentric_to_cm(xrel_ast, vrel_ast, masses)
        t, xva = integrate(x, v, masses, times)
        """
        times = jnp.arange(t_start, t_end, dt)
        t, xva = integrate_elements(elements, masses, times, t_start)

        tcarr = []
        for pidx in range(1, len(masses)):
            tc, gj = find_transit_times_nodata(t, xva, pidx, masses)
            tcarr.append(tc)

        if return_xva:
            return tcarr, t, xva

        return tcarr

    @partial(jit, static_argnums=(0,))
    def get_ttvs(self, elements, masses):
        """
        xrel_j, vrel_j = initialize_from_elements(elements, masses)
        xrel_ast, vrel_ast = jacobi_to_astrocentric(xrel_j, vrel_j, masses)
        x, v = astrocentric_to_cm(xrel_ast, vrel_ast, masses)
        t, xva = integrate(x, v, masses, self.times)
        """
        t, xva = integrate_elements(elements, masses, self.times, self.t_start)

        tcarr = jnp.array([])
        for pidx in range(1, len(masses)):
            tc, gj = find_transit_times(t, xva, pidx, self.tcobs[pidx-1], masses)
            tcarr = jnp.hstack([tcarr, tc])

        return tcarr

    def quicklook(self, model, sigma=None):
        data = self.tcobs_flatten
        for j in range(self.nplanet):
            idx = self.pidx==j+1
            plt.figure()
            plt.title("planet %d"%(j+1))
            plt.xlabel("transit time")
            plt.ylabel("TTV")
            if np.max(self.errorobs_flatten)==1:
                plt.plot(data[idx], (data - self.tcobs_linear)[idx], 'o', label='data')
            else:
                plt.errorbar(data[idx], (data - self.tcobs_linear)[idx], yerr=self.errorobs_flatten[idx], fmt='o', lw=1, label='data')
            if sigma is None:
                plt.plot(model[idx], (model - self.tcobs_linear)[idx], '.-', label='model', lw=1)
            else:
                m, s = (model - self.tcobs_linear)[idx], sigma[idx]
                plt.plot(model[idx], m, '.-', label='model', lw=1)
                plt.fill_between(model[idx], m-s, m+s, alpha=0.4, lw=1, color='C1')
            plt.legend(loc='best');



#%%
"""
dt = 0.1 * 0.5
t_start, t_end = 10, 1e4
jttv = jaxttv(t_start, t_end, dt)

#%%
elements = jnp.array([[365.25, 0.4-0.39, jnp.pi*0.5, 0.1*jnp.pi, 0.1*jnp.pi, 40], [365.25*2., 0., jnp.pi*0.5, 0.1*jnp.pi, 0.1*jnp.pi, 40]])#[:1]
masses = jnp.array([1, 300e-6, 300e-6])#[:2]

#%%
elements = jnp.array([[10, 0.1*jnp.cos(0.1*jnp.pi), 0.1*jnp.sin(0.1*jnp.pi), 0, 0, 2], [20.2, 0.2*jnp.cos(0.1*jnp.pi), 0.2*jnp.sin(0.1*jnp.pi), 0, 0, 3]])
masses = jnp.array([1, 30e-6, 30e-6])

#%%
tcobs, t, xva = jttv.get_ttvs_nodata(elements, masses, t_start, t_end, dt, return_xva=True)

#%%
np.shape(xva)
plt.xlim(t_start-1, 3*elements[0,0])
plt.plot(t, xva[:,xkey,1,2])
for _tc in tcobs[0][:5]:
    plt.axvline(x=_tc)

#%%
etot = get_energy_vmap(xva[:,xkey,:,:], xva[:,vkey,:,:], masses)
%timeit ediff = get_energy(xva[-1,xkey,:,:], xva[-1,vkey,:,:], masses) / get_energy(xva[0,xkey,:,:], xva[0,vkey,:,:], masses) - 1
ediff
plt.plot(t, (etot-etot[0])/etot[0], '-')

#%%
jttv.set_tcobs(tcobs, [elements[0][0], elements[1][0]])

#%%
tpars = jttv.get_ttvs(elements, masses)

#%%
jttv.quicklook(tpars)
"""

#%%
#from jax import grad
#def func(elements, masses):
#    return jnp.sum(jttv.get_ttvs(elements, masses))
#func(elements, masses)
#grad(func)(elements, masses)

#%%
#%time tpars = jttv.get_ttvs(elements, masses)
