
__all__ = ["compute_nbody_flux"]

import jax.numpy as jnp
from jax import jit, vmap
from exoplanet_core.jax import ops
rsun_au = 0.00465047

def get_xvast_map(xcm, vcm, pidxarr):
    def xvast_orbit(xcm, vcm, j):
        return xcm[j,:2] - xcm[0,:2], vcm[j,:2] - vcm[0,:2]
    xvast_map = vmap(xvast_orbit, (0,0,0), (0))
    return xvast_map(xcm, vcm, pidxarr)

def compute_relative_flux(barr, rarr, u1, u2):
    soln = ops.quad_solution_vector(barr, rarr)
    g = jnp.array([1.-u1-1.5*u2, u1+2*u2, -0.25*u2])
    I0 = jnp.pi * (g[0] + 2 * g[1] / 3.)
    return jnp.dot(soln, g) / I0 #- 1.

@jit
def compute_nbody_flux(rstar, prad, u1, u2, times, times_tidx, times_pidx, tc, xcm, vcm, pidxarr):
    xsky_tc, vsky_tc = get_xvast_map(xcm, vcm, pidxarr) # (Ntransit, xy)
    xsky_au = xsky_tc[times_tidx] + vsky_tc[times_tidx] * (times - tc[times_tidx])[:,None] # (Ntimes, xy)
    barr_au = jnp.sqrt(jnp.sum(xsky_au**2, axis=1)) # (Ntimes,)
    barr = barr_au / (rsun_au * rstar)
    rarr = prad[times_pidx]
    return compute_relative_flux(barr, rarr, u1, u2)
