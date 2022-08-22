""" symplectic integrator
much borrowed from TTVFast https://github.com/kdeck/TTVFast
"""

import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import scan
from .utils import *
from jax.config import config
config.update('jax_enable_x64', True)

#%%
def dEstep(x, ecosE0, esinE0, dM):
    """ single step to solve incremental Kepler's equation to obtain delta(eccentric anomaly)

        Args:
            x: initial guess for dE
            ecosE0, esinE0: eccentricity and eccentric anomaly at the initial state
            dM: delta(mean anomaly)

        Returns:
            delta(eccentric anomaly) from single iteration


    """
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
    """ Kepler step

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            gm: 'GM' in Kepler's 3rd law
            dt: time step
            nitr: number of iterations (currently needs to be fixed)

        Returns:
            new positions (Norbit, xyz)
            new velocities (Norbit, xyz)

    """
    r0 = jnp.sqrt(jnp.sum(x*x, axis=1))
    v0s = jnp.sum(v*v, axis=1)
    u = jnp.sum(x*v, axis=1)
    a = 1. / (2./r0 - v0s/gm)
    n = jnp.sqrt(gm / (a*a*a))
    ecosE0, esinE0 = 1. - r0 / a, u / (n*a*a)

    dM = n * dt
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

def Hint(x, v, masses):
    """ interaction Hamiltonian devided by Gm_0m_0

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            masses: masses of the bodies (Nbody,), solar unit

        Returns:
            value of interaction Hamiltonian

    """

    mu = masses[1:] / masses[0]

    ri = jnp.sqrt(jnp.sum(x * x, axis=1))
    Hint = jnp.sum(mu / ri)

    xast, vast = jacobi_to_astrocentric(x, v, masses)
    ri0 = jnp.sqrt(jnp.sum(xast * xast, axis=1))
    Hint -= jnp.sum(mu / ri0)

    xjk = jnp.transpose(xast[:,None] - xast[None, :], axes=[0,2,1])
    x2jk = jnp.sum(xjk * xjk, axis=1)
    nzidx = x2jk != 0.
    x2jk = jnp.where(nzidx, x2jk, 1.)
    xjkinv = jnp.where(nzidx, jnp.sqrt( 1. / x2jk ), 0.)
    Hint -= 0.5 * jnp.sum(mu[:,None] * mu[None,:] * xjkinv)

    return Hint

gHint = grad(Hint)

def Hintgrad(x, v, masses):
    """ gradient of the interaction Hamiltonian
    times (star mass / planet mass)

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            masses: masses of the bodies (Nbody,), solar unit

        Returns:
            gradient of interaction Hamiltonian x (star mass / planet mass)


    """
    return gHint(x, v, masses) * (masses[0] / masses[1:])[:,None]

#%%
#@jit
def nbody_kicks(x, v, ki, masses, dt):
    """ apply N-body kicks to velocities

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            ki: GM values
            masses: masses of the bodies (Nbody,), solar unit
            dt: time step

        Returns:
            positions
            kicked velocities

    """
    dv = - ki[:, None] * dt * Hintgrad(x, v, masses)
    return x, v+dv

#@jit
def symplectic_step(x, v, ki, masses, dt):
    """ advance the system by a single symplectic step
    (0.5dt kepler) -> (velocity kick) -> (0.5dt kepler again)

        Args:
            x: positions (Norbit, xyz)
            v: velocities (Norbit, xyz)
            ki: GM values
            masses: masses of the bodies (Nbody,), solar unit
            dt: time step

        Returns:
            new positions
            new velocities

    """
    dt2 = 0.5 * dt
    x, v = kepler_step(x, v, ki, dt2)
    x, v = nbody_kicks(x, v, ki, masses, dt)
    xout, vout = kepler_step(x, v, ki, dt2)
    return xout, vout

#@jit
def integrate_xv(x, v, masses, times):
    """ symplectic integration of the orbits

        Args:
            x: initial Jacobi positions (Norbit, xyz)
            v: initial Jacobi velocities (Norbit, xyz)
            masses: masses of the bodies (Nbody,), in units of solar mass
            times: cumulative sum of time steps

        Returns:
            times (initial time omitted)
            Jacobi position/velocity array (Nstep, x or v, Norbit, xyz)

    """
    ki = BIG_G * masses[0] * jnp.cumsum(masses)[1:] / jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])
    dtarr = jnp.diff(times)

    # transformation between the mapping and real Hamiltonian
    x, v = real_to_mapTO(x, v, ki, masses, dtarr[0])

    # single Kepler step (could be summarized as sympletic_step above)
    def step(xvin, dt):
        x, v = xvin
        dt2 = 0.5 * dt
        x, v = kepler_step(x, v, ki, dt2)
        x, v = nbody_kicks(x, v, ki, masses, dt)
        xout, vout = kepler_step(x, v, ki, dt2)
        return [xout, vout], jnp.array([xout, vout])

    _, xv = scan(step, [x, v], dtarr)

    return times[1:], xv

def compute_corrector_coefficientsTO():
    """ coefficients for the third-order corrector """
    corr_alpha = jnp.sqrt(7./40.)
    corr_beta = 1./(48.0*corr_alpha)

    TOa1, TOa2 = -corr_alpha, corr_alpha
    TOb1, TOb2 = -0.5 * corr_beta, 0.5 * corr_beta

    return TOa1, TOa2, TOb1, TOb2

def corrector_step(x, v, ki, masses, a, b):
    """ corrector step

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
    """ transformation between real and mapping coordinates

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

#%%
from functools import partial
from .hermite4 import find_transit_times, get_derivs # not used?
a2cm_map = vmap(astrocentric_to_cm, (0,0,None), 0)
geta_map = vmap(get_acm, (0,None), 0) 
j2a_map = vmap(jacobi_to_astrocentric, (0,0,None), 0) # not used?

#@jit
def xvjac_to_xvacm(xv, masses):
    """ Conversion from Jacobi to center-of-mass
    xv is assumed to be the result of integration: 1st axis is for the times.

        Args:
            xv: positions and velocities in Jacobi coordinates (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,), solar unit

        Returns:
            positions in the CoM frame (Nstep, Norbit)
            velocities in the CoM frame
            accelerations in the CoM frame

    """
    xa, va = jacobi_to_astrocentric(xv[:,0,:], xv[:,1,:], masses)
    xcm, vcm = a2cm_map(xa, va, masses)
    acm = geta_map(xcm, masses)
    return xcm, vcm, acm

#@jit
def find_transit_times_planets(t, x, v, a, tcobs, masses):
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
        tc = find_transit_times(t, x, v, a, j+1, tcobs[j], masses)
        tcarr = jnp.hstack([tcarr, tc])
    return tcarr

#@partial(jit, static_argnums=(0,))
def get_ttvs(self, elements, masses):
    """ compute model transit times given orbtal elements and masses

        Args:
            self: JaxTTV class
            elements: orbital elements in JaxTTV format
            masses: masses of the bodies (in units of solar mass)

        Returns:
            model transit times (1D flattened array)
            fractional energy change

    """
    x0, v0 = initialize_from_elements(elements, masses, self.t_start)
    t, xv = integrate_xv(x0, v0, masses, self.times)
    x, v, a = xvjac_to_xvacm(xv, masses)
    etot = get_energy_vmap(x, v, masses)
    tpars = find_transit_times_planets(t, x, v, a, self.tcobs, masses)
    return tpars, etot[-1]/etot[0]-1.
