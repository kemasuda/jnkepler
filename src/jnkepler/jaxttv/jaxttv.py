__all__ = ["Nbody", "JaxTTV", "plot_model", "get_means_and_stds"]


import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import warnings
from functools import partial
from .utils import *
from .conversion import *
from .findtransit import find_transit_times_single, find_transit_times_all, find_transit_times_kepler_all
from .symplectic import integrate_xv, kepler_step_map
from .hermite4 import integrate_xv as integrate_xv_hermite4
from .rv import *
from jax import jit, grad, config
config.update('jax_enable_x64', True)


class Nbody:
    """ superclass for nbody analysis """
    def __init__(self, t_start, t_end, dt, nitr_kepler=3):
        """ initialization

            Args:
                t_start: start time of integration
                t_end: end time of integration
                dt: integration time step (day)
                nitr_kepler: number of iterations in Kepler steps

        """
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.times = jnp.arange(t_start, t_end, dt)
        self.nitr_kepler = nitr_kepler


class JaxTTV(Nbody):
    """ main class for TTV analysis """
    def __init__(self, t_start, t_end, dt, nitr_kepler=3, transit_time_method='newton-raphson', nitr_transit=5):
        """ initialization

            Args:
                t_start: start time of integration
                t_end: end time of integration
                dt: integration time step (day)
                nitr_kepler: number of iterations in Kepler steps
                transit_time_method: Newton-Raphson or interpolation (latter not fully tested)
                nitr_transit: number of iterations in transit-finding loop (only for Newton-Raphson)

        """
        super(JaxTTV, self).__init__(t_start, t_end, dt, nitr_kepler=nitr_kepler)
        self.transit_time_method = transit_time_method
        self.nitr_transit = nitr_transit

    def set_tcobs(self, tcobs, p_init, errorobs=None, print_info=True):
        """ set observed transit times
        JaxTTV returns transit times that are closest to the observed times,
        rather than all the transit times between t_start and t_end

            Args:
                tcobs: list of the arrays of transit times for each planet
                p_init: initial guess for the mean orbital period of each planet
                errorobs: transit time error (currently assumed to be Gaussian), same format as tcobs

        """
        self.tcobs = tcobs
        self.tcobs_flatten = np.hstack([t for t in tcobs])
        self.nplanet = len(tcobs)
        #self.nbody = len(tcobs) + 1 # not used? may be confusing when nplanet_nt != 0
        self.p_init = p_init
        if errorobs is None:
            self.errorobs = None
            self.errorobs_flatten = jnp.ones_like(self.tcobs_flatten)
        else:
            self.errorobs = errorobs
            self.errorobs_flatten = np.hstack([e for e in errorobs])

        pidx, tcobs_linear, ttvamp = np.array([]), np.array([]), np.array([])
        for j in range(len(tcobs)):
            tc = tcobs[j]
            if len(tc) == 0: 
                continue
            elif len(tc) == 1:
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
            if np.nanmin(p_init)/self.dt < 20.:
                warnings.warn("time step may be too large.")
            print ()
            print ("# number of transiting planets:".ljust(37) + "%d"%self.nplanet)

        assert self.t_start < np.min(self.tcobs_flatten), "t_start seems too large compared to the first transit time in data."
        assert np.max(self.tcobs_flatten) < self.t_end, "t_end seems too small compared to the last transit time in data."

    def linear_ephemeris(self):
        """ (Re)derive linear ephemeris when necessary

            Returns:
                array of t0, array of P from linear fitting

        """
        p_new, t0_new = [], []

        for j in range(self.nplanet):
            tc = self.tcobs[j]
            m = np.round((tc - tc[0]) / self.p_init[j])
            if len(tc) > 1:
                pfit, t0fit = np.polyfit(m, tc, deg=1)
            else:
                pfit, t0fit = self.p_init[j], self.tcobs_linear[j]
                print ("# ephemeris of planet %d was not updated."%(j+1))
            t0_new.append(t0fit)
            p_new.append(pfit)

        return np.array(t0_new), np.array(p_new)

    @partial(jit, static_argnums=(0,))
    def get_ttvs(self, elements, masses):
        """ compute model transit times (jitted version)
        This function returns only transit times that are closest to the observed ones.
        To get all the transit times, use get_ttvs_nodata instead.

            Args:
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)

            Returns:
                1D flattened array of transit times
                fractional energy change

        """
        xjac0, vjac0 = initialize_jacobi_xv(elements, masses, self.t_start) # initial Jacobi position/velocity
        times, xvjac = integrate_xv(xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler) # integration
        orbit_idx = self.pidx.astype(int) - 1 # idx for orbit, starting from 0
        tcobs1d = self.tcobs_flatten # 1D array of observed transit times
        if self.transit_time_method == 'newton-raphson':
            transit_times = find_transit_times_all(orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_transit)
        else:
            transit_times = find_transit_times_kepler_all(orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_kepler)
        ediff = get_energy_diff_jac(xvjac, masses, -0.5*self.dt)
        return transit_times, ediff


    @partial(jit, static_argnums=(0,))
    def get_ttvs_and_rvs(self, elements, masses, times_rv):
        """ compute model transit times and stellar RVs

            Args:
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)
                times_rv: times at which stellar RVs are evaluated

            Returns:
                1D flattened array of transit times
                1D array of stellar RVs (m/s)
                fractional energy change

        """
        xjac0, vjac0 = initialize_jacobi_xv(elements, masses, self.t_start) # initial Jacobi position/velocity
        times, xvjac = integrate_xv(xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler) # integration
        orbit_idx = self.pidx.astype(int) - 1 # idx for orbit, starting from 0
        tcobs1d = self.tcobs_flatten # 1D array of observed transit times
        if self.transit_time_method == 'newton-raphson':
            transit_times = find_transit_times_all(orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_transit)
        else:
            transit_times = find_transit_times_kepler_all(orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_kepler)
        ediff = get_energy_diff_jac(xvjac, masses, -0.5*self.dt)

        nbodyrv = rv_from_xvjac(times_rv, times, xvjac, masses)

        return transit_times, nbodyrv, ediff

    def get_ttvs_nodata(self, elements, masses, t_start=None, t_end=None, dt=None, flatten=False,
        nitr_transit=5, nitr_kepler=3, symplectic=True, truncate=True):
        """ compute all model transit times between t_start and t_end
        This function is slower than get_ttvs and should not be used for fitting.

            Args:
                elements: orbital elements in JaxTTV format
                masses: masses of the bodies (in units of solar mass)
                t_start: beginning of integration
                t_end: end of integration
                dt: integration time step (day)
                flatten: if True, the returned transit time array is flattened
                nitr_transit: # of iterations in transit-search loop
                nitr_kepler: # of iterations in Kepler step (for symplectic only)
                symplectic: if True use symplectic; otherwise Hermite4 (needs smaller dt in general)
                truncate: if True, model transit times are truncated to fit inside the observing window of each planet

            Returns:
                list or 1D array (flatten=True) of model transit times
                fractional energy change

        """
        if (t_start is None) or (t_end is None) or (dt is None):
            times, t0, dt = self.times, self.t_start, self.dt
        else:
            times, t0 = jnp.arange(t_start, t_end, dt), t_start

        xjac0, vjac0 = initialize_jacobi_xv(elements, masses, t0)

        # note that the function for integration requires jit to avoid precision loss
        if symplectic:
            t, xcm, vcm, acm, de_frac = integrate_orbits_symplectic(xjac0, vjac0, masses, times, dt, nitr_kepler)
        else:
            t, xcm, vcm, acm, de_frac = integrate_orbits_hermite(xjac0, vjac0, masses, times)

        tcarr = []
        for pidx in range(1, len(self.tcobs)+1):
            tc = find_transit_times_single(t, xcm, vcm, acm, pidx, masses, nitr=nitr_transit)
            if truncate:
                t0lin, plin = self.tcobs_linear[pidx-1], self.p_init[pidx-1]
                epoch = np.round((tc - t0lin) / plin).astype(int)
                epochobs = np.round((self.tcobs[pidx-1] - t0lin) / plin).astype(int)
                emin, emax = np.min(epochobs), np.max(epochobs)
                idx = (emin <= epoch) & (epoch <= emax)
                tc = tc[idx]
            tcarr.append(tc)

        if flatten:
            tcarr = np.hstack(tcarr)

        return tcarr, de_frac

    def quicklook(self, model, sigma=None, save=None):
        """ plot observed and model TTVs (may be obsolete given plot_model)

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

    '''
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
            ax[1].set_ylabel('frequency (normalized)')
            x0 = np.linspace(-5, 5, 100)
            ax[1].plot(x0, np.exp(-0.5*x0**2/sd**2)/np.sqrt(2*np.pi)/sd, lw=1, color='C0', ls='dashed', label='$\mathrm{SD}=%.2f$ (jitter: %.1e)'%(sd,jitters[j]))
            ax[1].legend(loc='lower right')
    '''
    def check_residuals(self, tc, jitters=None, student=True, normalize_residuals=True, plot=True, fit_mean=False):
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
            res = self.tcobs_flatten[idx] - tc[idx]
            err0 = self.errorobs_flatten[idx]
            err = np.sqrt(err0**2 + jitters[j]**2)
            y = np.array(res / err)
            if not normalize_residuals:
                ax[0].errorbar(self.tcobs_flatten[idx], res*1440, yerr=self.errorobs_flatten[idx]*1440, fmt='o', lw=1,
                                label='SD=%.2e'%np.std(res))
                ax[0].set_ylabel('residual (min)')
            else:
                ax[0].errorbar(self.tcobs_flatten[idx], y, yerr=self.errorobs_flatten[idx]/err, fmt='o', lw=1,
                            label='$\chi^2=%.1f$ (%d data)'%(np.sum(y**2), len(y)))
                ax[0].set_ylabel('residual / assigned error')
            ax[0].set_title("planet %d"%(j+1))
            ax[0].set_xlabel("time (days)")
            ax[0].legend(loc='best')

            sd = np.std(y)
            rnds = np.random.randn(int(1e6))
            ax[1].set_yscale("log")
            ax[1].hist(y, histtype='step', lw=1, density=True, color='C0')
            ymin, ymax = ax[1].get_ylim()
            #ax[1].set_ylim(1e-3, ymax*1.5)
            ax[1].set_ylim(ymin/10, ymax*1.5)
            ax[1].set_title("planet %d"%(j+1))
            ax[1].set_xlabel("residual / assigned error")
            ax[1].set_ylabel('frequency (normalized)')
            x0 = np.linspace(-5, 5, 100)
            ax[1].plot(x0, np.exp(-0.5*x0**2/sd**2)/np.sqrt(2*np.pi)/sd, lw=1, color='C0', ls='dashed', label='$\mathrm{SD}=%.2f$ (jitter: %.1e)'%(sd,jitters[j]))
            ax[1].legend(loc='lower right')

        if not student:
            return None

        # Student's t fit
        res = self.tcobs_flatten - tc
        y = np.array(res / self.errorobs_flatten)
        #lnvar, lndf = fit_t_distribution(y, plot=plot, fit_mean=fit_mean)
        params_st = fit_t_distribution(y, plot=plot, fit_mean=fit_mean)

        return {'mean': np.mean(res), 'sd': np.std(res)}, params_st

    def check_timing_precision(self, params, dtfrac=1e-3, nitr_transit=10, nitr_kepler=10, symplectic=False):
        """ compare get_ttvs outputs with those from get_ttvs_nodata with a smaller timestep
        to check the precision of the former (may be obsolete)

            Args:
                params: JaxTTV parameter array
                dtfrac: (innermost period) * dtfrac is used for the comparison integration

            Returns:
                tc: model transit times from get_ttvs
                tc2: model transit times using a smaller timestep

        """
        elements, masses = params_to_elements(params, self.nplanet)
        tc, de = self.get_ttvs(elements, masses)
        print ("# fractional energy error (symplectic, dt=%.2e): %.2e" % (self.dt,de))

        dtcheck = self.p_init[0] * dtfrac
        tc2, de2 = self.get_ttvs_nodata(elements, masses, t_start=self.t_start, t_end=self.t_end, dt=dtcheck,
                                        flatten=True, nitr_transit=nitr_transit, nitr_kepler=nitr_kepler, symplectic=symplectic)
        intname = 'symplectic' if symplectic else 'Hermite4'
        print ("# fractional energy error (%s, dt=%.2e): %.2e" % (intname, dtcheck, de2))

        tc, tc2 = np.array(tc), np.array(tc2)
        tc2 = tc2[np.array(findidx_map(tc2, tc))]
        maxdiff = np.max(np.abs(tc-tc2))
        print ("# max difference in tc: %.2e days (%.2f sec)"%(maxdiff, maxdiff*86400))

        return tc, tc2

    def optim(self, dp=5e-1, dtic=1e-1, emax=0.5, mmin=1e-7, mmax=1e-3, cosilim=[-1e-6,1e-6], olim=[-1e-6,1e-6], amoeba=False, plot=True, save=None, pinit=None, jacrev=False, return_init=False):
        """ find maximum-likelihood parameters using scipy.optimize.curve_fit
        Could write a more elaborate function separately.

            Args:
                dp, dtic: search widths for periods and times of inferior conjunction
                emax: maximum allowed eccentricity
                mmin, mmax: minimum and maximum planet masses (in units of solar mass)
                cosilim, olim: bounds for cos(incliantion) and londitude of ascending node (radian)
                amoeba: if True, Nelder-Mead optimization is used
                plot: if True, show quicklook plots
                save: name of the plots to be saved (if not None)
                pinit: initial parameter array can be specified (if not None)
                jacrev: if True, jacobian from jax.grad is used for curve_fit

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

        if return_init:
            return pinit

        def getmodel(params):
            elements, masses = params_to_elements(params, npl)
            model = self.get_ttvs(elements, masses)[0]
            return model

        if jacrev and (not amoeba):
            from jax import jit, jacrev
            _jacfunc = jit(jacrev(getmodel)) # it may take long to compile...
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

        elements, masses = params_to_elements(pfinal, npl)
        if plot:
            #self.quicklook(getmodel(pfinal), save=save)
            t0_lin, p_lin = self.linear_ephemeris()
            tcall, _ = self.get_ttvs_nodata(elements, masses)
            plot_model(tcall, self.tcobs, self.errorobs, t0_lin, p_lin, marker='.', save=save)

        return pfinal, elements_to_pdic(elements, masses)

    def sample_means_and_stds(self, samples, N=50):
        """ compute mean and standard deviation of transit time models from HMC samples

            Args:
                samples: dictionary containing parameter samples (output of mcmc.get_samples())
                N: number of samples to be used for calculation

            Returns:
                means and standard deviations of transit time models, list of length(nplanet)

        """
        np.random.seed(123)
        sample_indices = np.random.randint(0, len(samples['masses']), N)
        models, means, stds = [], [], []
        for idx in sample_indices:
            elements, masses = samples['elements'][idx], samples['masses'][idx]
            models.append(self.get_ttvs_nodata(elements, masses)[0])
        means, stds = get_means_and_stds(models)
        return means, stds


@partial(jit, static_argnums=(5,))
def integrate_orbits_symplectic(xjac0, vjac0, masses, times, dt, nitr_kepler):
    """ symplectic integration of the orbits

        Args:
            xjac0: initial Jacobi positions (Norbit, xyz)
            vjac0: initial Jacobi velocities (Norbit, xyz)
            masses: masses of the bodies (in units of solar mass), (Nbody,)
            times: cumulative sum of time steps (note: step assumed to be constant)
            nitr_kepler: number of iterations in Kepler steps

        Returns:
            times, x/v/a in CM frame (Nstep, Norbit, xyz), fractional energy change

    """
    # symplectic integration
    t, xvjac = integrate_xv(xjac0, vjac0, masses, times, nitr=nitr_kepler)

    # account for the fact that output of integrate_xv is dt/2 ahead of the completion of the sympletic step
    dt_correct = -0.5 * dt
    de_frac = get_energy_diff_jac(xvjac, masses, dt_correct)
    t += dt_correct
    xjac, vjac = kepler_step_map(xvjac[:,0,:,:], xvjac[:,1,:,:], masses, dt_correct)

    # conversion to CM frame
    xcm, vcm, acm = xvjac_to_xvacm(xjac, vjac, masses)

    return t, xcm, vcm, acm, de_frac


@jit
def integrate_orbits_hermite(xjac0, vjac0, masses, times):
    """ symplectic integration of the orbits

        Args:
            xjac0: initial Jacobi positions (Norbit, xyz)
            vjac0: initial Jacobi velocities (Norbit, xyz)
            masses: masses of the bodies (in units of solar mass), (Nbody,)
            times: cumulative sum of time steps (note: step assumed to be constant)

        Returns:
            times, x/v/a in CM frame (Nstep, Norbit, xyz), fractional energy change

    """
    xast0, vast0 = jacobi_to_astrocentric(xjac0, vjac0, masses)
    xcm, vcm = astrocentric_to_cm(xast0, vast0, masses)
    t, xvacm = integrate_xv_hermite4(xcm, vcm, masses, times)
    xcm, vcm, acm = xvacm[:,0,:,:], xvacm[:,1,:,:], xvacm[:,2,:,:]
    de_frac = get_energy_diff(xvacm, masses)
    return t, xcm, vcm, acm, de_frac


def plot_model(tcmodellist, tcobslist, errorobslist, t0_lin, p_lin,
               tcmodelunclist=None, tmargin=None, save=None, marker=None, ylims=None, ylims_residual=None,
               unit=1440., ylabel='TTV (min)', xlabel='transit time (day)'):
    """ plot transit time model

        Args:
            tcmodellist: list of the arrays of model transit times for each planet
            tcobslist: list of the arrays of observed transit times for each planet
            errorobslist: list of the arrays of observed transit time errors for each planet
            t0_lin, p_lin: linear ephemeris used to show TTVs (n_planet,)
            tcmodelunclist: model uncertainty (same format as tcmodellist)
            tmargin: margin in x axis
            save: if not None, plot is saved as "save_planet#.png"
            marker: marker for model
            unit: TTV unit (defaults to minutes)
            ylabel, xlabel: axis labels in the plots
            ylims, ylims_residual: y ranges in the plots

    """
    for j, (tcmodel, tcobs, errorobs, t0, p) in enumerate(zip(tcmodellist, tcobslist, errorobslist, t0_lin, p_lin)):
        tcmodel, tcobs, errorobs = np.array(tcmodel), np.array(tcobs), np.array(errorobs)

        #plt.figure(figsize=(8,5))
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        if tmargin is not None:
            plt.xlim(np.min(tcobs)-tmargin, np.max(tcobs)+tmargin)
        ax.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        tnumobs = np.round((tcobs - t0)/p).astype(int)
        tnummodel = np.round((tcmodel - t0)/p).astype(int)
        ax.errorbar(tcobs, (tcobs-t0-tnumobs*p)*unit, yerr=errorobs*unit, zorder=1000,
                     fmt='o', mfc='white', color='dimgray', label='data', lw=1, markersize=7)
        #idxm = tcmodel < np.max(tcobs) + tmargin
        idxm = tcmodel > 0
        tlin = t0 + tnummodel * p
        ax.plot(tcmodel[idxm], (tcmodel-tlin)[idxm]*unit, '-', marker=marker, lw=1, mfc='white', color='steelblue',
                 zorder=-1000, label='model', alpha=0.9)
        if tcmodelunclist is not None:
            munc = tcmodelunclist[j]
            ax.fill_between(tcmodel[idxm], (tcmodel-munc-tlin)[idxm]*unit,
                            (tcmodel+munc-tlin)[idxm]*unit,
                             lw=1, color='steelblue', zorder=-1000, alpha=0.2)
        ax.set_title("planet %d"%(j+1))
        if ylims is not None and len(ylims)==len(t0_lin):
            ax2.set_ylim(ylims[j])

        idxm = findidx_map(tcmodel, tcobs) 
        ax2.errorbar(tcobs, (tcobs-tcmodel[idxm])*unit, yerr=errorobs*unit, zorder=1000,
                     fmt='o', mfc='white', color='dimgray', label='data', lw=1, markersize=7)
        ax2.axhline(y=0, color='steelblue', alpha=0.6)
        ax2.set_ylabel("residual (min)")
        if ylims_residual is not None and len(ylims_residual)==len(t0_lin):
            ax2.set_ylim(ylims_residual[j])

        # change legend order
        handles, labels = ax.get_legend_handles_labels()
        order = [1,0]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                   loc='upper left', bbox_to_anchor=(1,1))

        fig.tight_layout(pad=0.05)

        if save is not None:
            plt.savefig(save+"_planet%d.png"%(j+1), dpi=200, bbox_inches="tight")

def get_means_and_stds(models):
    """ get mean and standard deviation of the models

        Args:
            models: transit time models, (# of samples, # of planets, # of transits)

        Returns:
            means: mean of the models (# of planets, # of transits)

    """
    means, stds = [], []
    for j in range(len(models[0])):
        models_j = np.array([models[s][j] for s in range(len(models))])
        means.append(np.mean(models_j, axis=0))
        stds.append(np.std(models_j, axis=0))
    return means, stds

def fit_t_distribution(y, plot=True, fit_mean=False):
    from scipy.stats import t as tdist
    from scipy.stats import norm
    import numpyro
    import numpyro.distributions as dist
    import jax.random as random

    def model(y):
        logdf = numpyro.sample("lndf", dist.Uniform(jnp.log(0.1), jnp.log(100)))
        logvar = numpyro.sample("lnvar", dist.Uniform(-2, 10))
        df = numpyro.deterministic("df", jnp.exp(logdf))
        v1 = numpyro.deterministic("v1", jnp.exp(logvar))
        if fit_mean:
            mean = numpyro.sample("mean", dist.Uniform(-jnp.std(y), jnp.std(y)))
            numpyro.sample("obs", dist.StudentT(loc=mean, scale=jnp.sqrt(v1), df=df), obs=y)
        else:
            numpyro.sample("obs", dist.StudentT(scale=jnp.sqrt(v1), df=df), obs=y)

    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=500)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, y)
    mcmc.print_summary()

    samples = mcmc.get_samples()
    lndf, lnvar = np.mean(samples['lndf']), np.mean(samples['lnvar'])
    pout = {'lndf': lndf, 'lnvar': lnvar}
    if fit_mean:
        mean = np.mean(samples['mean'])
        pout['mean'] = mean
    else:
        mean = 0.

    if plot:
        sd = np.std(y)
        fig, ax = plt.subplots(1, 2, figsize=(16,4))
        ax[1].set_yscale("log")
        ax[1].set_ylabel("PDF")
        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("residual / assigned error")
        ax[1].set_xlabel("residual / assigned error")
        ax[1].hist(y, histtype='step', lw=3, alpha=0.6, density=True, color='gray')
        ymin, ymax = plt.gca().get_ylim()
        ax[1].set_ylim(ymin/5., ymax*1.5)
        x0 = np.linspace(-5, 5, 100)
        ax[1].plot(x0, norm(scale=sd).pdf(x0), lw=1, color='C0', ls='dashed', 
                label='normal, $\mathrm{SD}=%.2f$'%sd)
        ax[1].plot(x0, norm.pdf(x0), lw=1, color='C0', ls='dotted', 
                label='normal, $\mathrm{SD}=1$')
        ax[1].plot(x0, tdist(loc=mean, scale=np.exp(lnvar*0.5), df=np.exp(lndf)).pdf(x0), 
        label='Student\'s t\n(lndf=%.2f, lnvar=%.2f, mean=%.2f)'%(lndf, lnvar, mean))
        #ax[1].legend(loc='upper right', bbox_to_anchor=(1.5,1))

        #ax[0].hist(y, bins=len(y), histtype='step', lw=3, alpha=0.6, density=True, cumulative=True, color='red')
        ysum = np.ones_like(y)
        hist, edge = np.histogram(y, bins=len(y))
        ax[0].plot(np.r_[x0[0], edge[0], edge[:-1], edge[-1], x0[-1]], 
                np.r_[0, 0, np.cumsum(hist)/len(y), 1, 1], lw=3, alpha=0.6, color='gray')
        #ax[0].plot(np.sort(y), np.cumsum(ysum)/len(ysum), lw=3, alpha=0.6, color='gray')
        ax[0].plot(x0, norm(loc=0, scale=sd).cdf(x0), lw=1, color='C0', ls='dashed', 
                label='normal, $\mathrm{SD}=%.2f$'%sd)
        ax[0].plot(x0, norm.cdf(x0), lw=1, color='C0', ls='dotted', 
                label='normal, $\mathrm{SD}=1$')
        ax[0].plot(x0, tdist(loc=mean, scale=np.exp(lnvar*0.5), df=np.exp(lndf)).cdf(x0), 
        label='Student\'s t\n(lndf=%.2f, lnvar=%.2f, mean=%.2f)'%(lndf, lnvar, mean))
        ax[0].legend(loc='upper left', fontsize=14)

    return pout
