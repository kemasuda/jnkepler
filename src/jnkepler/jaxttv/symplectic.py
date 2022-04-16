#%%
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import scan
from .utils import *
from jax.config import config
config.update('jax_enable_x64', True)

#%%
#@jit
def dEstep(x, ecosE0, esinE0, dM):
    x2 = x / 2.0
    sx2, cx2 = jnp.sin(x2), jnp.cos(x2)
    sx, cx = 2.0*sx2*cx2, cx2*cx2 - sx2*sx2
    f = x + 2.0*sx2*(sx2*esinE0 - cx2*ecosE0) - dM
    ecosE = cx*ecosE0 - sx*esinE0
    fp = 1.0 - ecosE
    fpp = (sx*ecosE0 + cx*esinE0)/2.0
    fppp = ecosE/6.0
    dx = -f/fp
    dx = -f/(fp + dx*fpp)
    dx = -f/(fp + dx*(fpp + dx*fppp))
    return x + dx

#%%
#@jit
def kepler_step(x, v, gm, dt, nitr=3):
    r0 = jnp.sqrt(jnp.sum(x*x, axis=1))
    v0s = jnp.sum(v*v, axis=1)
    u = jnp.sum(x*v, axis=1)
    a = 1. / (2./r0 - v0s/gm)
    n = jnp.sqrt(gm / (a*a*a))
    ecosE0, esinE0 = 1. - r0 / a, u / (n*a*a)

    dM = n * dt
    """
    e0, E0 = jnp.sqrt(ecosE0*ecosE0 + esinE0*esinE0), jnp.arctan2(esinE0, ecosE0)
    M0 = E0 - esinE0
    E_new = get_E(reduce_angle(M0 + dM), e0)
    dE = E_new - E0
    """

    #dE = dEstep(dM, ecosE0, esinE0, dM)
    def step(x, i):
        return dEstep(x, ecosE0, esinE0, dM), None
    dE, _ = scan(step, dM, jnp.arange(nitr))

    x2 = dE / 2.
    sx2, cx2 = jnp.sin(x2), jnp.cos(x2)
    f = 1.0 - (a/r0)*2.0*sx2*sx2
    sx, cx = 2.0*sx2*cx2, cx2*cx2 - sx2*sx2
    g = (2.0*sx2*(esinE0*sx2 + cx2*r0/a))/n
    fp = 1.0 - cx*ecosE0 + sx*esinE0
    fdot = -(a/(r0*fp))*n*sx
    gdot = (1.0 + g*fdot)/f

    x_new = f[:,None] * x + g[:,None] * v
    v_new = fdot[:,None] * x + gdot[:,None] * v

    return x_new, v_new

""" something is wrong here
#@jit
def Hintgrad(x, v, masses):
    x2i = jnp.sum(x * x, axis=1)
    x3inv = 1. / (x2i * jnp.sqrt(x2i))
    a = - x * x3inv[:,None]

    xast, vast = jacobi_to_astrocentric(x, v, masses)
    nbody = len(masses)
    mmat = jnp.eye(nbody-1) + jnp.tril(jnp.tile(masses[1:] / jnp.cumsum(masses)[1:], (nbody-1,1)), k=-1)

    x2i = jnp.sum(xast * xast, axis=1)
    x3inv = 1. / (x2i * jnp.sqrt(x2i))
    a += jnp.dot(mmat.T, xast * x3inv[:,None])

    xjk = jnp.transpose(xast[:,None] - xast[None, :], axes=[0,2,1]) #j,xyz,k
    x2jk = jnp.sum(xjk * xjk, axis=1)[:,None,:] # j, None, k
    x2jk = jnp.where(x2jk!=0., x2jk, jnp.inf)
    x2jkinv = 1. / x2jk
    x2jkinv1p5 = x2jkinv * jnp.sqrt(x2jkinv)
    Xjk = xjk * x2jkinv1p5 # j, xyz, k
    mratio = (masses[1:][:,None] / masses[1:][None,:]).T
    mM = mratio * mmat.T
    a += jnp.dot(mM, jnp.dot(Xjk, masses[1:]/masses[0]))

    return a
"""

#%% interaction Hamiltonian devided by Gm_0m_0
def Hint(x, v, masses):
    mu = masses[1:] / masses[0]

    ri = jnp.sqrt(jnp.sum(x * x, axis=1))
    Hint = jnp.sum(mu / ri)

    xast, vast = jacobi_to_astrocentric(x, v, masses)
    ri0 = jnp.sqrt(jnp.sum(xast * xast, axis=1))
    Hint -= jnp.sum(mu / ri0)

    xjk = jnp.transpose(xast[:,None] - xast[None, :], axes=[0,2,1])
    x2jk = jnp.sum(xjk * xjk, axis=1)#[:,None,:]
    x2jk = jnp.where(x2jk!=0., x2jk, jnp.inf)
    xjkinv = jnp.sqrt( 1. / x2jk )
    Hint -= 0.5 * jnp.sum(mu[:,None] * mu[None,:] * xjkinv)

    return Hint

gHint = grad(Hint)

def Hintgrad(x, v, masses):
    return gHint(x, v, masses) *  (masses[0] / masses[1:])[:,None]


#%%
#@jit
def nbody_kicks(x, v, ki, masses, dt):
    dv = - ki[:, None] * dt * Hintgrad(x, v, masses)
    return x, v+dv

#@jit
def symplectic_step(x, v, ki, masses, dt):
    dt2 = 0.5 * dt
    x, v = kepler_step(x, v, ki, dt2)
    x, v = nbody_kicks(x, v, ki, masses, dt)
    xout, vout = kepler_step(x, v, ki, dt2)
    return xout, vout

#@jit
def integrate_xv(x, v, masses, times):
    ki = BIG_G * masses[0] * jnp.cumsum(masses)[1:] / jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])
    dtarr = jnp.diff(times)

    x, v = real_to_mapTO(x, v, ki, masses, dtarr[0])

    def step(xvin, dt):
        x, v = xvin
        dt2 = 0.5 * dt
        x, v = kepler_step(x, v, ki, dt2)
        x, v = nbody_kicks(x, v, ki, masses, dt)
        xout, vout = kepler_step(x, v, ki, dt2)
        #xout, vout = symplectic_step(x, v, ki, masses, dt)
        return [xout, vout], jnp.array([xout, vout])

    _, xv = scan(step, [x, v], dtarr)

    return times[1:], xv

#%%
#@jit
def compute_corrector_coefficientsTO():
    corr_alpha = jnp.sqrt(7./40.)
    corr_beta = 1./(48.0*corr_alpha)

    TOa1, TOa2 = -corr_alpha, corr_alpha
    TOb1, TOb2 = -0.5 * corr_beta, 0.5 * corr_beta

    return TOa1, TOa2, TOb1, TOb2

#@jit
def corrector_step(x, v, ki, masses, a, b):
    _x, _v = kepler_step(x, v, ki, -a)
    _x, _v = nbody_kicks(_x, _v, ki, masses, b)
    _x, _v = kepler_step(_x, _v, ki, a)
    return _x, _v

#@jit
def real_to_mapTO(x, v, ki, masses, dt):
    TOa1, TOa2, TOb1, TOb2 = compute_corrector_coefficientsTO()
    _x, _v = corrector_step(x, v, ki, masses, TOa2*dt, TOb2*dt)
    _x, _v = corrector_step(_x, _v, ki, masses, TOa1*dt, TOb1*dt)
    return _x, _v

#%%
#make_jaxpr(integrate_xv)(x0, v0, masses, times)
#from jax import make_jaxpr
#make_jaxpr(real_to_mapTO)(x0, v0, ki, masses, 0.01)

#%%
from functools import partial
from .hermite4 import find_transit_times, get_derivs
#from hermite4 import  get_derivs, get_gderivs
#a2cm_map = jit(vmap(astrocentric_to_cm, (0,0,None), 0))
#geta_map = jit(vmap(get_derivs, (0,0,None), 0))
#j2a_map = jit(vmap(jacobi_to_astrocentric, (0,0,None), 0))
a2cm_map = vmap(astrocentric_to_cm, (0,0,None), 0)
#geta_map = vmap(get_derivs, (0,0,None), 0)
geta_map = vmap(get_acm, (0,None), 0)
j2a_map = vmap(jacobi_to_astrocentric, (0,0,None), 0)

#@jit
def xvjac_to_xvacm(xv, masses):
    xa, va = jacobi_to_astrocentric(xv[:,0,:], xv[:,1,:], masses)
    xcm, vcm = a2cm_map(xa, va, masses)
    #acm, _ = geta_map(xcm, vcm, masses)
    acm = geta_map(xcm, masses)
    return xcm, vcm, acm
    #xva = jnp.array([xcm, vcm, acm]).transpose(1,0,2,3)
    #return xva

#@jit
def find_transit_times_planets(t, x, v, a, tcobs, masses):
    #x, v, a = xvjac_to_xvacm(xv, masses)
    tcarr = jnp.array([])
    for j in range(len(masses)-1):
        tc = find_transit_times(t, x, v, a, j+1, tcobs[j], masses)
        tcarr = jnp.hstack([tcarr, tc])
    return tcarr

#@partial(jit, static_argnums=(0,))
def get_ttvs(self, elements, masses):
    x0, v0 = initialize_from_elements(elements, masses, self.t_start)
    t, xv = integrate_xv(x0, v0, masses, self.times)
    x, v, a = xvjac_to_xvacm(xv, masses)
    etot = get_energy_vmap(x, v, masses)
    tpars = find_transit_times_planets(t, x, v, a, self.tcobs, masses)
    return tpars, etot[-1]/etot[0]-1.

#%%
"""
step_vmap = jit(vmap(symplectic_step, (0,0,None,None,0), 2)) # xva, body idx, xyz, transit idx
#@jit
def find_transit_times(t, xvjac, x, v, a, j, tcobs, masses, nitr=5):
    ki = BIG_G * masses[0] * jnp.cumsum(masses)[1:] / jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])
    xastj, vastj, aastj = cm_to_astrocentric(x, v, a, j)
    gj, dotgj = get_gderivs(xastj, vastj, aastj)

    # reasonable
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
    xjtc = xvjac[1:,0,:,:][idxsort][tcidx]
    vjtc = xvjac[1:,1,:,:][idxsort][tcidx]

    def tcstep(xvs, i):
        xin, vin, step = xvs
        xj, vj = step_vmap(xin, vin, ki, masses, step)
        xj = jnp.transpose(xj, axes=[2,0,1])
        vj = jnp.transpose(vj, axes=[2,0,1])
        xa, va = j2a_map(xj, vj, masses)
        xcm, vcm = a2cm_map(xa, va, masses)
        acm, _ = geta_map(xcm, vcm, masses)
        _xastj, _vastj, _aastj = cm_to_astrocentric(xcm, vcm, acm, j)
        _gj, _dotgj = get_gderivs(_xastj, _vastj, _aastj)
        step = - _gj / _dotgj
        return [xj, vj, step], step

    _, steps = scan(tcstep, [xjtc, vjtc, nrstep], jnp.arange(nitr))
    tc += nrstep + jnp.sum(steps, axis=0)

    return tc

#@jit
def find_transit_times_planets(t, xv, x, v, a, tcobs, masses):
    #x, v, a = xvjac_to_xvacm(xv, masses)
    tcarr = jnp.array([])
    for j in range(len(masses)-1):
        tc = find_transit_times(t, xv, x, v, a, j+1, tcobs[j], masses)
        tcarr = jnp.hstack([tcarr, tc])
    return tcarr

@partial(jit, static_argnums=(0,))
def get_ttvs(self, elements, masses):
    x0, v0 = initialize_from_elements(elements, masses, self.t_start)
    t, xv = integrate_xv(x0, v0, masses, self.times)
    x, v, a = xvjac_to_xvacm(xv, masses)
    etot = get_energy_vmap(x, v, masses)
    tpars = find_transit_times_planets(t, xv, x, v, a, self.tcobs, masses)
    return tpars, etot[-1]/etot[0]-1.
"""

#%%
"""
from jax import make_jaxpr
import numpy as np
import matplotlib.pyplot as plt
from jaxttv import jaxttv

#%%
elements = jnp.array([[365.25, 0.4*jnp.cos(0.1*jnp.pi), 0.4*jnp.sin(0.1*jnp.pi), 0, 0.1*jnp.pi, 40], [365.25*2., 0., 0, 0, 0.1*jnp.pi, 40]])
masses = jnp.array([1, 300e-6, 300e-6])

#%%
elements = jnp.array([[10, 0.1*jnp.cos(0.1*jnp.pi), 0.1*jnp.sin(0.1*jnp.pi), 0, 0, 2], [20.2, 0.2*jnp.cos(0.1*jnp.pi), 0.2*jnp.sin(0.1*jnp.pi), 0, 0, 3]])
masses = jnp.array([1, 30e-6, 30e-6])

#%%
elements = jnp.array([[10, 0.1*jnp.cos(0.1*jnp.pi), 0.1*jnp.sin(0.1*jnp.pi), 0, 0, 2], [20.2, 0.2*jnp.cos(0.1*jnp.pi), 0.2*jnp.sin(0.1*jnp.pi), 0, 0, 3], [30, 0.1*jnp.cos(-0.5*jnp.pi), 0.1*jnp.sin(-0.5*jnp.pi), 0, 0, 4]])
masses = jnp.array([1, 3e-6, 3e-6, 1e-6])

#%%
dt = 0.25
t_start, t_end = 0, 4 * 365.25
times = jnp.arange(t_start, t_end, dt)

#%%
jttv = jaxttv(t_start, t_end, dt)
tcobs, defrac = jttv.get_ttvs_nojit(elements, masses, t_start, t_end, 0.001)
print (defrac)

#%%
p_init = [jnp.mean(jnp.diff(tcobs[j])) for j in range(len(tcobs))]
jttv.set_tcobs(tcobs, p_init)

#%%
x0, v0 = initialize_from_elements(elements, masses, t_start)

#%%
#from hermite4 import get_derivs
#make_jaxpr(get_derivs)(x0, v0, masses[1:])
#make_jaxpr(Hintgrad)(x0, v0, masses)

#%%
#%timeit t, xv = integrate_xv(x0, v0, masses, times)
t, xv = integrate_xv(x0, v0, masses, times)

#%%
x, v, a = xvjac_to_xvacm(xv, masses)

#%%
%timeit etot = get_energy_vmap(x, v, masses)
etot = get_energy_vmap(x, v, masses)
#%timeit ediff = get_ediff(jnp.array([x,v,a]).transpose(1,0,2,3), masses)
#make_jaxpr(get_energy_vmap)(x, v, masses)

#%%
plt.figure()
plt.plot(t, etot/etot[0] - 1., '-', label='$\Delta E/E = %.2e$'%(etot[-1]/etot[0]-1.))
plt.legend();

#%%
%timeit tpars = find_transit_times_planets(t, x, v, a, tcobs, masses)
tpars = find_transit_times_planets(t, x, v, a, tcobs, masses)
make_jaxpr(find_transit_times_planets)(t, x, v, a, tcobs, masses)

#%%
%timeit tpars, ediff = get_ttvs(jttv, elements, masses)
print (ediff)

#%%
plt.xlabel("transit time difference (sec)")
plt.hist(np.array(tpars - jnp.hstack(tcobs))*86400)

#%%
jttv.quicklook(tpars)

#%%
func = lambda elements, masses: jnp.sum(get_ttvs(elements, masses)[0])
gfunc = jit(grad(func))
%timeit func(elements, masses)
%timeit gfunc(elements, masses)
"""
