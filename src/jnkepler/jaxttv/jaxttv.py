__all__ = ["JaxTTV"]

#%%
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
from .utils import *
from .conversion import *
from .findtransit import *
from .symplectic import integrate_xv
from .hermite4 import integrate_xv as integrate_xv_hermite4
from jax import jit, grad
from jax.config import config
config.update('jax_enable_x64', True)

#%%
class JaxTTV:
    """ main class """
    def __init__(self, t_start, t_end, dt):
        """ initialization

            Args:
                t_start: start time of integration
                t_end: end time of integration
                dt: integration time step (day)

        """
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.times = jnp.arange(t_start, t_end, dt)

    def set_tcobs(self, tcobs, p_init, errorobs=None, print_info=True):
        """ set observed transit times
        JaxTTV returns transit times that are closest to the observed times,
        rather than all the transit times between t_start and t_end

            Args:
                tcobs: list of the arrays of transit times for each planet
                p_init: initial guess for the mean orbital period of each planet
                errorobs: transit time error (currently assumed to be Gaussian),same format as tcobs

        """
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
            if len(tc)==0:
                continue
            elif len(tc)==1:
                pidx = np.r_[pidx, np.ones_like(tc)*(j+1)]
                tc_linear = tc[0]
                tcobs_linear = np.r_[tcobs_linear, tc_linear]
            else:
                pidx = np.r_[pidx, np.ones_like(tc)*(j+1)]
                m = np.round((tc - tc[0]) / p_init[j])
                pfit, t0fit = np.polyfit(m, tc, deg=1)
                tc_linear = t0fit + m*pfit
                tcobs_linear = np.r_[tcobs_linear, tc_linear]

            ttv = tc - tc_linear
            ttvamp = np.r_[ttvamp, np.max(ttv)-np.min(ttv)]

        self.pidx = pidx
        self.tcobs_linear = tcobs_linear
        self.ttvamp = ttvamp

        if print_info:
            print ("# integration starts at:".ljust(35) + "%.2f"%self.t_start)
            print ("# first transit time in data:".ljust(35) + "%.2f"%np.min(self.tcobs_flatten))
            print ("# last transit time in data:".ljust(35) + "%.2f"%np.max(self.tcobs_flatten))
            print ("# integration ends at:".ljust(35) + "%.2f"%self.t_end)
            print ("# integration time step:".ljust(35) + "%.4f (1/%d of innermost period)"%(self.dt, np.nanmin(p_init)/self.dt))

    def update_period(self):
        """ Re-derive linear ephemeris if necessary

            Returns:
                array of t0, array of P from linear fitting

        """
        p_new, t0_new = [], []
        for j in range(self.nplanet):
            tc = self.tcobs[j]
            m = np.round((tc - tc[0]) / self.p_init[j])
            pfit, t0fit = np.polyfit(m, tc, deg=1)
            t0_new.append(t0fit)
            p_new.append(pfit)
        return np.array(t0fit), np.array(p_new)

    """
    def integrate(self, elements, masses, t_start, t_end, dt):
        integrate the orbits using Hermite integrator

            Args:
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)
                t_start: beginning of integration
                t_end: end of integration
                dt: integration time step (day)

            Returns:
                t: array of times
                xva: position (x), velocity (v), acceleration (a)
                    shape = (time, x or v or a, orbit, cartesian component)
                energy: total energy at each time

        times = jnp.arange(t_start, t_end, dt)
        t, xva = integrate_elements(elements, masses, times, t_start)
        energy = get_energy_vmap(xva[:,0,:,:], xva[:,1,:,:], masses)
        return t, xva, energy
    """

    @partial(jit, static_argnums=(0,))
    def get_ttvs(self, elements, masses, nitr_kepler=3, nitr_transit=5):
        """ compute model transit times (jitted version)
        Returns only transit times that are closest to the observed times.

            Args:
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)

            Returns:
                1D flattened array of transit times
                fractional energy change

        """
        xjac0, vjac0 = initialize_jacobi_xv(elements, masses, self.t_start)
        times, xvjac = integrate_xv(xjac0, vjac0, masses, self.times, nitr=nitr_kepler)
        #xcm, vcm, acm = xvjac_to_xvacm(xvjac, masses)
        #etot = get_energy_map(xcm, vcm, masses)
        #transit_times = find_transit_times_planets(times, xcm, vcm, acm, self.tcobs, masses, nitr=nitr_transit)
        #transit_times = find_transit_times_all(self.pidx.astype(int)-1, self.tcobs_flatten, times, xcm, vcm, acm, masses, nitr=nitr_transit)
        transit_times, etot = find_transit_times_all(self.pidx.astype(int)-1, self.tcobs_flatten, times, xvjac, masses, nitr=nitr_transit)
        return transit_times, etot[-1]/etot[0]-1.

    def get_ttvs_nodata(self, elements, masses, t_start=None, t_end=None, dt=None, flatten=False,
        nitr_transit=5):
        """ compute all model transit times between t_start and t_end
        This function is much slower than get_ttvs and should not be used for fitting
        Now Hermite4 integration is used (but this is not necessary).

            Args:
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)
                t_start: beginning of integration
                t_end: end of integration
                dt: integration time step (day)
                flatten: if True, the returned transit time array is flattened
                nitr: # of iterations in transit-search loop

            Returns:
                list or 1D array (flatten=True) of model transit times
                fractional energy change

        """
        if t_start is not None:
            times, t0 = jnp.arange(t_start, t_end, dt), t_start
        else:
            times, t0 = self.times, self.t_start

        xjac0, vjac0 = initialize_jacobi_xv(elements, masses, t0)
        xast0, vast0 = jacobi_to_astrocentric(xjac0, vjac0, masses)
        xcm, vcm = astrocentric_to_cm(xast0, vast0, masses)
        t, xva = integrate_xv_hermite4(xcm, vcm, masses, times)
        de_frac = get_energy_diff(xva, masses)

        tcarr = []
        for pidx in range(1, len(masses)):
            tc = find_transit_times_nodata(t, xva[:,0,:,:], xva[:,1,:,:], xva[:,2,:,:], pidx, masses, nitr=nitr_transit)
            tcarr.append(tc)
        if flatten:
            tcarr = np.hstack(tcarr)

        return tcarr, de_frac

    def quicklook(self, model, sigma=None, save=None):
        """ plot observed and model TTVs

            Args:
                model: model transit times (1D flattened)
                sigma: model uncertainty (e.g. SD of posterior models)
                save: name of the plots when they are saved

        """
        data = self.tcobs_flatten
        for j in range(self.nplanet):
            idx = self.pidx==j+1
            if not np.sum(idx):
                continue
            plt.figure()
            plt.title("planet %d"%(j+1))
            plt.xlabel("transit time")
            plt.ylabel("TTV")
            if np.max(self.errorobs_flatten)==1:
                plt.plot(data[idx], (data - self.tcobs_linear)[idx], 'o', label='data', mfc='none')
            else:
                plt.errorbar(data[idx], (data - self.tcobs_linear)[idx], yerr=self.errorobs_flatten[idx], fmt='o', lw=1, label='data', mfc='none')
            if sigma is None:
                plt.plot(model[idx], (model - self.tcobs_linear)[idx], '.-', label='model', lw=1)
            else:
                m, s = (model - self.tcobs_linear)[idx], sigma[idx]
                plt.plot(model[idx], m, '.-', label='model', lw=1)
                plt.fill_between(model[idx], m-s, m+s, alpha=0.4, lw=1, color='C1')
            plt.legend(loc='best')
            if save is not None:
                plt.savefig(save+"%d.png"%(j+1), dpi=200, bbox_inches="tight")

    def check_residuals(self, tc, jitters=None):
        """ plot residuals from a given model
        Compare the histogram of O-C with Gaussians.

            Args:
                model: model transit times (1D flattened)
                jitters: may be added to errorobs_flatten

        """
        if jitters is not None:
            jitters = np.atleast_1d(jitters)
            if len(jitters) == 1:
                jitters = np.array([jitters[0]] * self.nplanet)
            else:
                assert len(jitters) == self.nplanet
        else:
            jitters = np.zeros(self.nplanet)

        for j in range(self.nplanet):
            fig, ax = plt.subplots(1,2,figsize=(16,4))
            idx = self.pidx==j+1
            res = self.tcobs_flatten[idx]-tc[idx]
            err0 = self.errorobs_flatten[idx]
            err = np.sqrt(err0**2 + jitters[j]**2)
            ax[0].errorbar(self.tcobs_flatten[idx], res*1440, yerr=self.errorobs_flatten[idx]*1440, fmt='o', lw=1,
                          label='SD=%.2e'%np.std(res))
            ax[0].set_title("planet %d"%(j+1))
            ax[0].set_xlabel("time (days)")
            ax[0].set_ylabel('residual (min)')
            ax[0].legend(loc='best')

            sd = np.std(np.array(res/err))
            rnds = np.random.randn(int(1e6))
            ax[1].set_yscale("log")
            ax[1].hist(np.array(res/err), histtype='step', lw=1, density=True, color='C0')
            ymin, ymax = ax[1].get_ylim()
            ax[1].set_ylim(1e-3, ymax*1.5)
            ax[1].set_title("planet %d"%(j+1))
            ax[1].set_xlabel("residual / error")
            #ymin, ymax = ax[1].get_ylim()
            #ax[1].set_ylim(1e-4, ymax*1.5)
            ax[1].set_ylabel('frequency (normalized)')
            x0 = np.linspace(-5, 5, 100)
            ax[1].plot(x0, np.exp(-0.5*x0**2/sd**2)/np.sqrt(2*np.pi)/sd, lw=1, color='C0', ls='dashed', label='$\mathrm{SD}=%.2f$ (jitter: %.1e)'%(sd,jitters[j]))
            ax[1].legend(loc='lower right')

    def check_prec(self, params, dtfrac=1e-3, nitr_transit=10):
        """ compare get_ttvs outputs with those from get_ttvs_data with small timestep
        to check the precision of the former (may be obsolete)

            Args:
                params: JaxTTV parameter array
                dtfrac: (innermost period) * dtfrac is used for the comparison integration

            Returns:
                model transit times from get_ttvs

        """
        elements, masses = params_to_elements(params, self.nplanet)
        tc, de = self.get_ttvs(elements, masses)
        print ("# fractional energy error (symplectic, dt=%.2e): %.2e" % (self.dt,de))

        dtcheck = self.p_init[0] * dtfrac
        tc2, de2 = self.get_ttvs_nodata(elements, masses, t_start=self.t_start, t_end=self.t_end, dt=dtcheck,
                                        flatten=True, nitr_transit=nitr_transit)
        tc2 = tc2[np.array(findidx_map(tc2, tc))]
        print ("# fractional energy error (Hermite, dt=%.2e): %.2e" % (dtcheck, de2))
        maxdiff = np.max(np.abs(tc-tc2))
        print ("# max difference in tc: %.2e days (%.2f sec)"%(maxdiff, maxdiff*86400))

        return tc

    def optim(self, dp=5e-1, dtic=1e-1, emax=0.5, mmin=1e-7, mmax=1e-3, cosilim=[-1e-6,1e-6], olim=[-1e-6,1e-6], amoeba=False, plot=True, save=None, pinit=None, jacrev=False):
        """ find maximum-likelihood parameters

            Returns:
                set of parameters (JaxTTV format)

        """
        from scipy.optimize import curve_fit
        import time

        npl = self.nplanet

        params_lower, params_upper, pnames = [], [], []
        for j in range(npl): # need to be changed to take into account non-transiting planets
            params_lower += [self.p_init[j]-dp, -emax, -emax, cosilim[0], olim[0], self.tcobs[j][0]-dtic]
            params_upper += [self.p_init[j]+dp, emax+1e-2, emax+1e-2, cosilim[1], olim[1], self.tcobs[j][0]+dtic]
            pnames += ["p%d"%(j+1), "ec%d"%(j+1), "es%d"%(j+1), "cosi%d"%(j+1), "om%d"%(j+1), "tic%d"%(j+1)]
        params_lower += [jnp.log(mmin)] * npl
        params_upper += [jnp.log(mmax)] * npl
        pnames += ["m%d"%(j+1) for j in range(npl)]
        params_lower = jnp.array(params_lower).ravel()
        params_upper = jnp.array(params_upper).ravel()

        bounds = (params_lower, params_upper)
        if pinit is None:
            pinit = 0.5 * (params_lower + params_upper)

        def getmodel(params):
            elements, masses = params_to_elements(params, npl)
            model = self.get_ttvs(elements, masses)[0]
            return model

        if jacrev and (not amoeba):
            from jax import jit, jacrev
            _jacfunc = jit(jacrev(getmodel)) # it takes long to compile...
            def jacfunc(x, *params):
                return _jacfunc(jnp.array(params))
            jac = jacfunc
        else:
            jac = None

        objective = lambda params: jnp.sum(((self.tcobs_flatten - getmodel(params)) / self.errorobs_flatten)**2)
        print ("initial objective function: %.2f (%d data)"%(objective(pinit), len(self.tcobs_flatten)))

        start_time = time.time()
        if amoeba:
            from scipy.optimize import minimize
            print ()
            print ("running Nelder-Mead optimization...")
            res = minimize(objective, pinit, bounds=np.array(bounds).T, method='Nelder-Mead', options={'adaptive': True})
            print ("objective function: %.2f (%d data)"%(objective(res.x), len(self.tcobs_flatten)))
            pinit = res.x

        func = lambda x, *params: getmodel(np.array(params))
        print ()
        print ("running optimization...")
        popt, pcov = curve_fit(func, None, self.tcobs_flatten, p0=pinit, sigma=self.errorobs_flatten, bounds=bounds, jac=jac)
        print ("objective function: %.2f (%d data)"%(objective(popt), len(self.tcobs_flatten)))
        print ("# elapsed time (least square): %.1f sec" % (time.time()-start_time))
        pfinal = popt

        if plot:
            self.quicklook(getmodel(pfinal), save=save)

        return pfinal
