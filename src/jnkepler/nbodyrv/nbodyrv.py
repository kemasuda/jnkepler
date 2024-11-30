
__all__ = ["NbodyRV"]

from ..jaxttv import Nbody
from ..jaxttv.rv import *
from ..jaxttv.utils import initialize_jacobi_xv
from ..jaxttv.symplectic import integrate_xv


class NbodyRV(Nbody):
    """class for RV-only analysis"""

    def __init__(self, t_start, t_end, dt, nitr_kepler=3):
        """initialization

            Args:
                t_start: start time of integration
                t_end: end time of integration
                dt: integration time step (day)
                nitr_kepler: number of iterations in Kepler steps

        """
        super(NbodyRV, self).__init__(
            t_start, t_end, dt, nitr_kepler=nitr_kepler)

    def get_rv(self, times_rv, par_dict):
        """compute stellar RV

            Args:
                times_rv: times at which RVs are evaluated
                par_dict: dictionary of input parameters (same as TTVs)

            Returns:
                array: stellar RVs at times_rvs (m/s), positive when the star is moving away

        """
        xjac0, vjac0, masses = initialize_jacobi_xv(
            par_dict, self.t_start)
        times, xvjac = integrate_xv(
            xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler)  # integration
        return rv_from_xvjac(times_rv, times, xvjac, masses)
