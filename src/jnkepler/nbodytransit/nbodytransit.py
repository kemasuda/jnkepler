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
    """ main class for photodynamical analysis """

    def set_lcobs(self, times_lc, overlapping_transit=False, exposure_time=29.4/1440., supersample_factor=10, print_info=True):
        """ initialization

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
        """ compute sky-plane positions and velocities at transit centers

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
        return tc, xsky_tc, vsky_tc

    @partial(jit, static_argnums=(0,))
    def get_lc(self, par_dict, rstar, prad, u1, u2):
        """ compute nbody flux

            Args:
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)
                rstar: stellar radius (in units of solar radius)
                prad: planet-to-star radius ratio (Norbit,)
                u1, u2: quadratic limb-darkening coefficients

            Returns:
                nbodyflux: transit light curve (len(times_lc),)
                tc: transit times (1D flattened array)

        """
        tc, xsky_tc, vsky_tc = self.get_xvsky_tc(par_dict)

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
    def get_lc_and_rv(self, times_rv, par_dict, rstar, prad, u1, u2):
        """ compute nbody flux and RV

            Args:
                times_rv: times at which RVs are evaluated
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)
                rstar: stellar radius (in units of solar radius)
                prad: planet-to-star radius ratio (Norbit,)
                u1, u2: quadratic limb-darkening coefficients

            Returns:
                nbodyflux: transit light curve (len(times_lc),)
                tc: transit times (1D flattened array)
                nbodyrv: stellar RVs at times_rvs (m/s), positive when the star is moving away

        """
        tc, xsky_tc, vsky_tc = self.get_xvsky_tc(par_dict)

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

    def optimize_transit_parameters(self, fluxes, errors, par_dict, rstar_init=1.,
                                    method="TNC", n_iter=1, pradmax=0.2):
        """ optimize parameters relevant to light curves alone

            Args:
                fluxes: observed flux
                errors: error of the fluxes
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)
                rstar_init: initial guess for stellar radius (in units of solar radius)
                method: method for jaxopt.ScipyBoundedMinimize
                n_iter: # of iterations in each optimization (unnecessary?)
                pradmax: maximum planet-to-star radius ratio

            Returns:
                a set of optimal parameters

        """
        import jaxopt
        npl = self.nplanet
        zeros = np.zeros(npl)
        ones = np.ones(npl)
        p_init = {
            "rstar": np.float64(rstar_init),
            "prad": ones * 0.01,
            "q1": np.float64(0.5),
            "q2": np.float64(0.5),
            "b": ones * 0.1,
        }

        def objective(p):
            u1, u2 = q_to_u(p['q1'], p['q2'])
            par_dict['cosi'] = b_to_cosi(
                p['b'], par_dict['period'], par_dict['ecosw'], par_dict['esinw'], p['rstar'])
            model = self.get_lc(par_dict, p['rstar'], p['prad'], u1, u2)[0]
            return jnp.sum(((fluxes - model) / errors)**2)

        solver = jaxopt.ScipyBoundedMinimize(fun=objective, method=method)
        print("# initial objective function:", objective(p_init))
        print()

        # optimize radii
        p_low = {"rstar": 0., "prad": zeros,
                 "q1": 0, "q2": 0, "b": p_init['b']}
        p_upp = {"rstar": p_init['rstar']*2, "prad": ones *
                 pradmax, "q1": 1, "q2": 1, "b": p_init['b']}
        bounds = (p_low, p_upp)
        print("# optimizing radius ratios...")
        for i in range(n_iter):
            res = solver.run(p_init, bounds=bounds)
            p_init, state = res
            print(state)
            print()

        # optimize impact parameters
        p_low = {"rstar": p_init['rstar'],
                 "prad": p_init['prad'], "q1": 0, "q2": 0, "b": zeros}
        p_upp = {"rstar": p_init['rstar'],
                 "prad": p_init['prad'], "q1": 1, "q2": 1, "b": ones}
        bounds = (p_low, p_upp)
        print("# optimizing impact parameters...")
        for i in range(n_iter):
            res = solver.run(p_init, bounds=bounds)
            p_init, state = res
            print(state)
            print()

        # optimize all
        p_low = {"rstar": 0., "prad": zeros, "q1": 0, "q2": 0, "b": zeros}
        p_upp = {"rstar": p_init['rstar']*2.,
                 "prad": ones*pradmax, "q1": 1, "q2": 1, "b": ones}
        bounds = (p_low, p_upp)
        print("# optimizing all parameters...")
        for i in range(n_iter):
            res = solver.run(p_init, bounds=bounds)
            p_init, state = res
            print(state)
            print()

        return p_init


def q_to_u(q1, q2):
    """ convert q1, q2 into u1, u2

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
    """ convert b into cosi following Eq.7 of Winn (2010), arXiv:1001.2010

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
