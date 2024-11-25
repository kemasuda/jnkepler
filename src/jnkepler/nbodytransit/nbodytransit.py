from ..jaxttv.rv import *
from .transit import *
__all__ = ["NbodyTransit", "q_to_u", "b_to_cosi"]

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
from ..jaxttv import JaxTTV
from ..jaxttv.utils import *
from ..jaxttv.conversion import *
from ..jaxttv.findtransit import *
from ..jaxttv.symplectic import integrate_xv, kepler_step_map
from ..jaxttv.hermite4 import integrate_xv as integrate_xv_hermite4
from jax import jit, grad, config
config.update('jax_enable_x64', True)


class NbodyTransit(JaxTTV):
    """main class for photodynamical analysis

        Unlike in JaxTTV class, non-transiting objects are not yet supported.

    """

    def set_lcobs(self, times_lc, overlapping_transit=False, exposure_time=29.4/1440., supersample_factor=10, print_info=True):
        """initialization

            Args:
                times_lc: times in the light curve
                overlapping_transit: if True, the code accounts for simultaneous transits; overlap between the planets is not yet supported
                exposure_time: exposure time (same unit as times_lc)
                supersample_factor: flux is computed at intervals given by exposure_time / supersample_factor and then summed
                print_info: if True, show the parameters

        """
        self.times_lc = times_lc

        self.supersample_num = int(supersample_factor // 2 * 2 + 1)
        if self.supersample_num < 2:
            self.times_super = times_lc
        else:
            dt = exposure_time / (self.supersample_num - 1)
            dtarr = np.linspace(0, exposure_time, self.supersample_num)
            dtarr -= np.median(dtarr)
            self.times_super = (times_lc[:, None] + dtarr).ravel()

        times_transit_idx, times_planet_idx = [], []
        for j in range(self.nplanet):
            tcj = np.where(self.pidx == j+1, self.tcobs_flatten, -np.inf)
            _tidx = findidx_map(tcj, self.times_super)
            _pidx = jnp.ones_like(_tidx) * j
            times_transit_idx.append(_tidx)
            times_planet_idx.append(_pidx)
        self.times_transit_idx = jnp.array(times_transit_idx)
        self.times_planet_idx = jnp.array(times_planet_idx)
        self.times_transit_idx_nool = findidx_map(
            self.tcobs_flatten, self.times_super)
        self.times_planet_idx_nool = self.pidx[self.times_transit_idx_nool].astype(
            int) - 1
        self.overlapping_transit = overlapping_transit

        if print_info:
            print("# exposure time (min):".ljust(
                35) + "%.1f" % (exposure_time*1440))
            print("# supersampling factor:".ljust(35) + "%d" %
                  (self.supersample_num-1))
            if not self.overlapping_transit:
                print("# overlapping transit ignored.".ljust(35))

    def get_xvsky_tc(self, par_dict):
        """compute sky-plane positions and velocities at transit centers

            Args:
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)

            Returns:
                tc: transit centers (Ntransit)
                xsky_tc: astrocentric positions in the sky plane at transit centers (Ntransit, xy)
                vsky_tc: astrocentric velocities in the sky plane at transit centers (Ntransit, xy)

        """
        xjac0, vjac0, masses = initialize_jacobi_xv(
            par_dict, self.t_start)  # initial Jacobi position/velocity
        times, xvjac = integrate_xv(
            xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler)  # integration
        pidxarr = self.pidx.astype(int)  # idx for planet, starting from 1
        tcobs1d = self.tcobs_flatten
        tc, (xcm, vcm, _) = find_transit_params_all(
            pidxarr-1, tcobs1d, times, xvjac, masses)
        xsky_tc, vsky_tc = get_xvast_map(xcm, vcm, pidxarr)
        return tc, xsky_tc, vsky_tc, times, xvjac, masses

    @partial(jit, static_argnums=(0,))
    def get_flux(self, par_dict):
        """compute nbody flux

            Args:
                par_dict: dict of input parameters; TTV parameters + 
                    srad: stellar radius (in units of solar radius)
                    radius_ratio: planet-to-star radius ratio (Norbit,)
                    u1, u2: quadratic limb-darkening coefficients

            Returns:
                nbodyflux: transit light curve (len(times_lc),)
                tc: transit times (1D flattened array)

        """
        tc, xsky_tc, vsky_tc, _, _, _ = self.get_xvsky_tc(par_dict)
        rstar, prad, u1, u2 = initialize_transit_params(par_dict)

        if self.overlapping_transit:
            nbodyflux_ss = compute_nbody_flux(
                rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, self.times_super, self.times_transit_idx, self.times_planet_idx)
        else:
            nbodyflux_ss = compute_nbody_flux_nooverlap(
                rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, self.times_super, self.times_transit_idx_nool, self.times_planet_idx_nool)

        nbodyflux = jnp.mean(
            nbodyflux_ss.reshape(-1, self.supersample_num), axis=1)

        return nbodyflux, tc

    @partial(jit, static_argnums=(0,))
    def get_flux_and_rv(self, par_dict, times_rv):
        """compute nbody flux and RV

            Args:
                par_dict: dict of input parameters; TTV parameters + 
                    srad: stellar radius (in units of solar radius)
                    radius_ratio: planet-to-star radius ratio (Norbit,)
                    u1, u2: quadratic limb-darkening coefficients
                times_rv: times at which RVs are evaluated

            Returns:
                nbodyflux: transit light curve (len(times_lc),)
                tc: transit times (1D flattened array)
                nbodyrv: stellar RVs at times_rvs (m/s), positive when the star is moving away

        """
        tc, xsky_tc, vsky_tc, times, xvjac, masses = self.get_xvsky_tc(
            par_dict)
        rstar, prad, u1, u2 = initialize_transit_params(par_dict)

        if self.overlapping_transit:
            nbodyflux_ss = compute_nbody_flux(
                rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, self.times_super, self.times_transit_idx, self.times_planet_idx)
        else:
            nbodyflux_ss = compute_nbody_flux_nooverlap(
                rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, self.times_super, self.times_transit_idx_nool, self.times_planet_idx_nool)

        nbodyflux = jnp.mean(
            nbodyflux_ss.reshape(-1, self.supersample_num), axis=1)
        nbodyrv = rv_from_xvjac(times_rv, times, xvjac, masses)

        return nbodyflux, tc, nbodyrv


def initialize_transit_params(par_dict):
    """initialize transit parameters

        Args:
            par_dict

        Returns:
            stellar radius (solar unit); stellar mass = 1 assumed if only density is specified in par_dict
            radius ratio
            coefficients for quadratic limb-darkening law

    """
    keys = par_dict.keys()

    if "smass" in keys and "srad" in keys:
        srad = par_dict["srad"]
    elif "sdens" in keys:
        smass = par_dict["smass"] if "smass" in keys else 1.
        srad = (smass / par_dict["sdens"])**(1./3.)
    else:
        raise ValueError(
            "Either (smass, srad) (stellar mass/radius in solar unit) or sdens (stellar mean density in solar unit) needs to be provided.")

    if "q1" in keys and "q2" in keys:
        u1, u2 = q_to_u(par_dict["q1"], par_dict["q2"])
    elif "u1" in keys and "u2" in keys:
        u1, u2 = par_dict["u1"], par_dict["u2"]
    else:
        raise ValueError(
            "Either (q1, q2) or (u1, u2) needs to be provided for quadratic limb-darkening.")

    if "radius_ratio" in keys:
        prad = par_dict["radius_ratio"]
    else:
        raise ValueError(
            "radius_ratio needs to be provided for planet-to-star radius ratios.")

    return srad, prad, u1, u2


def q_to_u(q1, q2):
    """convert q1, q2 into u1, u2

        Args:
            q1, q2: quadratic limb-darkening coefficients as parameterized in Kipping, D. M. 2013, MNRAS, 435, 2152

        Returns:
            u1, u2: quadratic limb-darkening coefficients

    """
    usum = jnp.sqrt(q1)
    u1 = 2 * q2 * usum
    u2 = usum - u1
    return u1, u2


def b_to_cosi(b, period, ecosw, esinw, rstar, mstar=1.):
    """convert b into cosi following Eq.7 of Winn (2010), arXiv:1001.2010

        Args:
            b: impact parameter (normalized to stellar radius)
            period: orbital period
            ecosw: eccentricity * cos(argument of periastron)
            esinw: eccentricity * sin(argument of periastron)
            rstar: stellar radius (solar unit)
            mstar: stellar mass (solar unit)

        Returns:
            cosine of inclination

    """
    a_over_r = 4.2083 * period**(2./3.) / rstar * mstar**(1. /
                                                          3.)  # adopting G, M_sun, R_sun from astropy.constants
    efactor = (1. - ecosw**2 - esinw**2) / (1. + esinw)
    return b / a_over_r / efactor
