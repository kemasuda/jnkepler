
__all__ = ["rv_from_xvjac"]

import jax.numpy as jnp
from ..jaxttv.conversion import jacobi_to_astrocentric, a2cm_map
au_per_day = 1.495978707e11 / 86400. # m/s

def rv_from_xvjac(times_rv, times, xvjac, masses):
    """ compute stellar RV
    NOTE: This will be more efficient if interpolation is performed before conversion to CM frame

        Args:
            times_rv: times at which RVs are evaluated
            times: times for n-body integration (Ntime,)
            xvjac: jacobi positions and velocities (Ntime, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,), solar unit

        Returns:
            stellar RVs at times_rvs (m/s), positive when the star is moving away

    """
    xast, vast = jacobi_to_astrocentric(xvjac[:,0,:,:], xvjac[:,1,:,:], masses)
    xcm, vcm = a2cm_map(xast, vast, masses)
    return - jnp.interp(times_rv, times, vcm[:,0,2]) * au_per_day


class NbodyRV:
    """ class for the RV-only analysis """
    def __init__(self):
        pass

    def get_rv(self, times_rv, elements, masses):
        xjac0, vjac0 = initialize_jacobi_xv(elements, masses, self.t_start) # initial Jacobi position/velocity
        times, xvjac = integrate_xv(xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler) # integration
        return rv_from_xvjac(times_rv, times, xvjac, masses)
