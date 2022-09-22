__all__ = ["NbodyTransit", "q_to_u", "b_to_cosi"]

#%%
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
from jax import jit, grad
from jax.config import config
config.update('jax_enable_x64', True)

#%%
from .transit import *
from ..nbodyrv.nbodyrv import *
class NbodyTransit(JaxTTV):
    """ main class for the photodynamical analysis """
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
            self.times_super = (times_lc[:,None] + dtarr).ravel()

        times_transit_idx, times_planet_idx = [], []
        for j in range(self.nplanet):
            tcj = np.where(self.pidx==j+1, self.tcobs_flatten, -np.inf)
            _tidx = findidx_map(tcj, times_lc)
            _pidx = jnp.ones_like(_tidx) * j
            times_transit_idx.append(_tidx)
            times_planet_idx.append(_pidx)
        self.times_transit_idx = jnp.array(times_transit_idx)
        self.times_planet_idx = jnp.array(times_planet_idx)
        self.times_transit_idx_nool = findidx_map(self.tcobs_flatten, self.times_super)
        self.times_planet_idx_nool = self.pidx[self.times_transit_idx_nool].astype(int) - 1
        self.overlapping_transit = overlapping_transit

        if print_info:
            print ("# exposure time (min):".ljust(35) + "%.1f"%(exposure_time*1440))
            print ("# supersampling factor:".ljust(35) + "%d"%(self.supersample_num-1))
            if not self.overlapping_transit:
                print ("# overlapping transit ignored.".ljust(35))

    @partial(jit, static_argnums=(0,))
    def get_lc(self, elements, masses, rstar, prad, u1, u2):
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
        xjac0, vjac0 = initialize_jacobi_xv(elements, masses, self.t_start) # initial Jacobi position/velocity
        times, xvjac = integrate_xv(xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler) # integration
        pidxarr = self.pidx.astype(int) # idx for planet, starting from 1
        tcobs1d = self.tcobs_flatten
        tc, (xcm, vcm, _) = find_transit_params_all(pidxarr-1, tcobs1d, times, xvjac, masses)
        xsky_tc, vsky_tc = get_xvast_map(xcm, vcm, pidxarr)

        if self.overlapping_transit:
            nbodyflux_ss = compute_nbody_flux(rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, self.times_super, self.times_transit_idx, self.times_planet_idx) * 0
        else:
            nbodyflux_ss = compute_nbody_flux_nooverlap(rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, self.times_super, self.times_transit_idx_nool, self.times_planet_idx_nool)

        nbodyflux = jnp.mean(nbodyflux_ss.reshape(-1, self.supersample_num), axis=1)

        return nbodyflux, tc

    @partial(jit, static_argnums=(0,1,))
    def get_lc_and_rv(self, times_rv, elements, masses, rstar, prad, u1, u2):
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
        xjac0, vjac0 = initialize_jacobi_xv(elements, masses, self.t_start) # initial Jacobi position/velocity
        times, xvjac = integrate_xv(xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler) # integration
        pidxarr = self.pidx.astype(int) # idx for planet, starting from 1
        tcobs1d = self.tcobs_flatten
        tc, (xcm, vcm, _) = find_transit_params_all(pidxarr-1, tcobs1d, times, xvjac, masses)
        xsky_tc, vsky_tc = get_xvast_map(xcm, vcm, pidxarr)

        if self.overlapping_transit:
            nbodyflux_ss = compute_nbody_flux(rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, self.times_super, self.times_transit_idx, self.times_planet_idx) * 0
        else:
            nbodyflux_ss = compute_nbody_flux_nooverlap(rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, self.times_super, self.times_transit_idx_nool, self.times_planet_idx_nool)

        nbodyflux = jnp.mean(nbodyflux_ss.reshape(-1, self.supersample_num), axis=1)
        nbodyrv = rv_from_xvjac(times_rv, times, xvjac, masses)

        return nbodyflux, tc, nbodyrv

    def optimize_transit_parameters(self, fluxes, errors, elements, masses, rstar_init=1.,
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
        zeros = np.zeros(len(masses)-1)
        ones = np.ones(len(masses)-1)
        p_init = {
            "rstar": np.float64(rstar_init),
            "prad": ones * 0.01,
            "q1": np.float64(0.5),
            "q2": np.float64(0.5),
            "b": ones * 0.1,
        }

        def objective(p):
            u1, u2 = q_to_u(p['q1'], p['q2'])
            cosi = b_to_cosi(p['b'], elements[:,0], elements[:,1], elements[:,2], p['rstar'])
            _elements = jnp.c_[elements[:,:3], cosi, elements[:,4:]]
            model = self.get_lc(_elements, masses, p['rstar'], p['prad'], u1, u2)[0]
            return jnp.sum(((fluxes - model) / errors)**2)

        solver = jaxopt.ScipyBoundedMinimize(fun=objective, method=method)
        print ("# initial objective function:", objective(p_init))
        print ()

        # optimize radii
        p_low = {"rstar": 0., "prad": zeros, "q1": 0, "q2": 0, "b": p_init['b']}
        p_upp = {"rstar": p_init['rstar']*2, "prad": ones*pradmax, "q1": 1, "q2": 1, "b": p_init['b']}
        bounds = (p_low, p_upp)
        print ("# optimizing radius ratios...")
        for i in range(n_iter):
            res = solver.run(p_init, bounds=bounds)
            p_init, state = res
            print (state)
            print ()

        # optimize impact parameters
        p_low = {"rstar": p_init['rstar'], "prad": p_init['prad'], "q1": 0, "q2": 0, "b": zeros}
        p_upp = {"rstar": p_init['rstar'], "prad": p_init['prad'], "q1": 1, "q2": 1, "b": ones}
        bounds = (p_low, p_upp)
        print ("# optimizing impact parameters...")
        for i in range(n_iter):
            res = solver.run(p_init, bounds=bounds)
            p_init, state = res
            print (state)
            print ()

        # optimize all
        p_low = {"rstar": 0., "prad": zeros, "q1": 0, "q2": 0, "b": zeros}
        p_upp = {"rstar": p_init['rstar']*2., "prad": ones*pradmax, "q1": 1, "q2": 1, "b": ones}
        bounds = (p_low, p_upp)
        print ("# optimizing all parameters...")
        for i in range(n_iter):
            res = solver.run(p_init, bounds=bounds)
            p_init, state = res
            print (state)
            print ()

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
    a_over_r = 3.7528 * period**(2./3.) / rstar * mstar**(1./3.)
    efactor = (1. - ecosw**2 - esinw**2) / (1. + esinw)
    return b / a_over_r / efactor

#%%
"""
import pandas as pd
d = pd.read_csv("/Users/k_masuda/Dropbox/repos/jnkepler/examples/kep51/ttv.txt", delim_whitespace=True, header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
tcobs = [jnp.array(d.tc[d.planum==j+1]) for j in range(3)]
errorobs = [jnp.array(d.tcerr[d.planum==j+1]) for j in range(3)]
p_init = [45.155305, 85.31646, 130.17809]

#%%
dt = 1.0
t_start, t_end = 155., 2950.
nt = NbodyTransit(t_start, t_end, dt)
nt.set_tcobs(tcobs, p_init, errorobs=errorobs, print_info=True)

#%%
params_best = nt.optim(mmax=1e-4, emax=0.1)

#%%
elements, masses = params_to_elements(params_best, nt.nplanet)
prad = jnp.array([0.08, 0.09, 0.1])
rstar = 1.
u1, u2 = 0.5, 0.2

#%%
data = pd.read_csv("lc.txt", delim_whitespace=True, header=None, names=['time', 'flux', 'flux_err', 'tranum', 'plnum'])
df = data.sort_values("time").reset_index(drop=True)

#%%
times_lc = jnp.array(df.time)
fluxes, errors = jnp.array(df.flux), jnp.array(df.flux_err)

#%%
nt.set_lcobs(times_lc, overlapping_transit=False, supersample_factor=-2.)

#%%
nt.supersample_num
nt.times_super
nt.times_lc
nt.times_super

#%%
%timeit model = nt.get_lc(elements, masses, rstar, prad, u1, u2)

#%%
plt.figure(figsize=(14,6))
plt.xlim(203.5, 205)
plt.plot(nt.times_lc, model, '.')

#%%
plt.figure(figsize=(14,6))
plt.xlim(203.5, 205)
plt.plot(nt.times_lc, model, '.')

#%%
exposure_time = 29.4 / 1440.
supersample_factor = 10
dt = exposure_time / supersample_factor

#%%
dtarr = np.linspace(0, exposure_time, int(supersample_factor // 2 * 2 + 1))
dtarr -= np.median(dtarr)

#%%
times_super = (times_lc[:,None] + dtarr).ravel()

#%%
nt.set_lcobs(times_super, overlapping_transit=False)

#%%
%timeit model = nt.get_lc(elements, masses, rstar, prad, u1, u2)

#%%
a = np.linspace(0, 10, 10)
a
a.reshape(-1, 5)

#%%
model_sum = jnp.mean(model.reshape(-1, len(dtarr)), axis=1)

#%%
plt.figure(figsize=(14,6))
plt.xlim(203.5, 205)
plt.plot(times_super, model, '.')
plt.plot(times_lc, model_sum, 'o')
plt.plot(times_lc, model_noss, 's')

#%%
nt.set_lcobs(times_lc, overlapping_transit=False)
model_noss = nt.get_lc(elements, masses, rstar, prad, u1, u2)

#%%
plt.plot(nt.times_lc, model)
"""
