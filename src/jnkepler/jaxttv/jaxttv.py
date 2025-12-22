"""Main class for modeling TTVs."""
__all__ = ["Nbody", "JaxTTV"]


import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import warnings
from functools import partial
from copy import deepcopy
from .utils import *
from .conversion import *
from .findtransit import find_transit_times_all, find_transit_times_kepler_all
from .symplectic import integrate_xv, kepler_step_map
from .hermite4 import integrate_xv as integrate_xv_hermite4
from .rv import *
from ..infer import fit_t_distribution
from jax import jit, grad, config
config.update('jax_enable_x64', True)


class Nbody:
    """superclass for nbody analysis"""

    def __init__(self, t_start, t_end, dt, nitr_kepler=3):
        """initialization

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
    """main class for TTV analysis"""

    def __init__(self, t_start, t_end, dt, tcobs, p_init, errorobs=None, print_info=True, nitr_kepler=3, transit_time_method='newton-raphson', nitr_transit=5):
        """initialization

            Args:
                t_start: start time of integration
                t_end: end time of integration
                dt: integration time step (day)
                nitr_kepler: number of iterations in Kepler steps
                transit_time_method: Newton-Raphson or interpolation (latter not fully tested)
                nitr_transit: number of iterations in transit-finding loop (only for Newton-Raphson)

            Attributes:
                transit_time_method: algorithm for transit time computation 
                nit_transit: number of Newton-Raphson iterations in computing transit times
                nplanet: number of transiting planets
                p_init: initial guess for mean orbital periods for tracking transit epochs
                tcobs: list of observed transit time arrays
                tcobs_flatten: 1D flattend version of tcobs
                errorobs: list of timing error arrays
                errorobs_flatten: 1D flattend version of errorobs
                pidx: index to specify planet in tcobs_flatten and errorobs_flatten (starting from 1)
                tcobs_linear: list of arrays of linear transit times calculated from the first transit time and p_init

        """
        super(JaxTTV, self).__init__(
            t_start, t_end, dt, nitr_kepler=nitr_kepler)

        tcobs, tcobs_flatten, nplanet, p_init, errorobs, errorobs_flatten, pidx, tcobs_linear \
            = self.set_tcobs(tcobs, p_init, errorobs=errorobs, print_info=print_info)

        if transit_time_method != "newton-raphson":
            warnings.warn(
                "transit_time_method other than newton-raphson is not well tested.")
        self.transit_time_method = transit_time_method
        self.nitr_transit = nitr_transit
        self.nplanet = nplanet
        self.p_init = np.asarray(p_init)
        self.tcobs = tcobs
        self.tcobs_flatten = tcobs_flatten
        self.errorobs = errorobs
        self.errorobs_flatten = errorobs_flatten
        self.pidx = pidx
        self.tcobs_linear = tcobs_linear

    def set_tcobs(self, tcobs, p_init, errorobs=None, print_info=True):
        """set observed transit times

            Note:
                JaxTTV returns transit times that are closest to the observed times set here, rather than all the transit times between t_start and t_end.

            Args:
                tcobs: list of the arrays of transit times for each planet
                p_init: initial guess for the mean orbital period of each planet
                errorobs: transit time error (currently assumed to be Gaussian), same format as tcobs

            Returns:
                attributes of JaxTTV class

        """
        tcobs = [np.array(t) for t in tcobs]
        tcobs_flatten = np.hstack([t for t in tcobs])
        nplanet = len(tcobs)
        if errorobs is None:
            errorobs_flatten = jnp.ones_like(tcobs_flatten)
        else:
            errorobs_flatten = np.hstack([e for e in errorobs])

        pidx, tcobs_linear = np.array([]), np.array([])
        for j in range(len(tcobs)):
            tc = tcobs[j]
            if len(tc) == 0:
                raise ValueError("Elements of tcobs should have length > 0.")
            elif len(tc) == 1:
                pidx = np.r_[pidx, np.ones_like(tc)*(j+1)]
                tc_linear = tc[0]
                tcobs_linear = np.r_[tcobs_linear, tc_linear]
            else:
                pidx = np.r_[pidx, np.ones_like(tc)*(j+1)]
                m = np.round((tc - tc[0]) / p_init[j])
                pfit, t0fit = np.polyfit(m, tc, deg=1)
                tc_linear = t0fit + m * pfit
                tcobs_linear = np.r_[tcobs_linear, tc_linear]

        if print_info:
            print("# number of transiting planets:".ljust(35) + "%d" % nplanet)
            print("# integration starts at:".ljust(35) + "%.2f" % self.t_start)
            print("# first transit time in data:".ljust(
                35) + "%.2f" % np.min(tcobs_flatten))
            print("# last transit time in data:".ljust(
                35) + "%.2f" % np.max(tcobs_flatten))
            print("# integration ends at:".ljust(35) + "%.2f" % self.t_end)
            print("# integration time step:".ljust(
                35) + "%.4f (1/%d of innermost period)" % (self.dt, np.nanmin(p_init)/self.dt))
            if np.nanmin(p_init) / self.dt < 20.:
                warnings.warn("time step may be too large.")
            print()

        assert self.t_start < np.min(
            tcobs_flatten), "t_start seems too large compared to the first transit time in data."
        assert np.max(
            tcobs_flatten) < self.t_end, "t_end seems too small compared to the last transit time in data."

        return tcobs, tcobs_flatten, nplanet, p_init, errorobs, errorobs_flatten, pidx, tcobs_linear

    def linear_ephemeris(self):
        """(Re)derive linear ephemeris when necessary

            Returns:
                tuple:
                    - array of t0 from linear fitting
                    - array of P from linear fitting

        """
        p_new, t0_new = [], []

        for j in range(self.nplanet):
            tc = self.tcobs[j]
            m = np.round((tc - tc[0]) / self.p_init[j])
            if len(tc) > 1:
                pfit, t0fit = np.polyfit(m, tc, deg=1)
            else:
                pfit, t0fit = self.p_init[j], self.tcobs_linear[j]
                warnings.warn(
                    "ephemeris of planet %d was not updated." % (j+1))
            t0_new.append(t0fit)
            p_new.append(pfit)

        return np.array(t0_new), np.array(p_new)

    @partial(jit, static_argnums=(0,))
    def get_transit_times_obs(self, par_dict, transit_orbit_idx=None):
        """compute model transit times 

            Note:
                This function returns only transit times that are closest to the observed ones.
                To get all the transit times, use get_transit_times_all instead.

            Args:
                par_dict: dict containing parameters
                transit_orbit_idx: array of idx of the planet (orbit) for each transit times should be evaulated, starting from 0. \
                    This must be specified when non-transiting planets exist. \
                    If None, all the planets are assumed to be transiting; this is equivalent to setting transit_orbit_idx = jnp.arange(nplanet).

            Returns:
                tuple:
                    - 1D array: flattened array of transit times
                    - float: fractional energy change

        """
        xjac0, vjac0, masses = initialize_jacobi_xv(
            par_dict, self.t_start)  # initial Jacobi position/velocity
        times, xvjac = integrate_xv(
            xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler)  # integration
        # idx for orbit, starting from 0
        if transit_orbit_idx is None:
            orbit_idx = self.pidx.astype(int) - 1
        else:
            orbit_idx = transit_orbit_idx[self.pidx.astype(
                int) - 1].astype(int)
        tcobs1d = self.tcobs_flatten  # 1D array of observed transit times
        if self.transit_time_method == 'newton-raphson':
            transit_times = find_transit_times_all(
                orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_transit)
        else:
            transit_times = find_transit_times_kepler_all(
                orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_kepler)
        ediff = get_energy_diff_jac(xvjac, masses, -0.5*self.dt)
        return transit_times, ediff

    @partial(jit, static_argnums=(0,))
    def get_transit_times_and_rvs_obs(self, par_dict, times_rv, transit_orbit_idx=None):
        """compute model transit times and stellar RVs

            Args:
                par_dict: dict containing parameters
                times_rv: times at which stellar RVs are evaluated
                transit_orbit_idx: array of idx of the planet (orbit) for each transit times should be evaulated, starting from 0. \
                    This must be specified when non-transiting planets exist. \
                    If None, all the planets are assumed to be transiting; this is equivalent to setting transit_orbit_idx = jnp.arange(nplanet).

            Returns:
                tuple:
                    - 1D array: flattened array of transit times
                    - 1D array: stellar RVs (m/s)
                    - float: fractional energy change

        """
        xjac0, vjac0, masses = initialize_jacobi_xv(
            par_dict, self.t_start)  # initial Jacobi position/velocity
        times, xvjac = integrate_xv(
            xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler)  # integration
        # idx for orbit, starting from 0
        if transit_orbit_idx is None:
            orbit_idx = self.pidx.astype(int) - 1
        else:
            orbit_idx = transit_orbit_idx[self.pidx.astype(
                int) - 1].astype(int)
        tcobs1d = self.tcobs_flatten  # 1D array of observed transit times
        if self.transit_time_method == 'newton-raphson':
            transit_times = find_transit_times_all(
                orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_transit)
        else:
            transit_times = find_transit_times_kepler_all(
                orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_kepler)
        ediff = get_energy_diff_jac(xvjac, masses, -0.5*self.dt)

        nbodyrv = rv_from_xvjac(times_rv, times, xvjac, masses)

        return transit_times, nbodyrv, ediff

    def tcall_linear(self, t_start, t_end, truncate=False):
        """information on all linear transit times between t_start and t_end

            Args:
                t_start: start time of integration
                t_end: end time of integration

            Returns:
                tuple:
                    - 1D array of orbit (planet) index starting from 1
                    - list of linear transit times between t_start and t_end
                    - 1D flattend version of tcall_linear

        """
        tcall_linear = []
        pidxall = np.array([])

        for j in range(len(self.tcobs)):
            tc = self.tcobs[j]
            if len(tc) == 1:
                t0fit, pfit = tc[0], self.p_init[j]
            else:
                m = np.round((tc - tc[0]) / self.p_init[j])
                pfit, t0fit = np.polyfit(m, tc, deg=1)

            m_min, m_max = np.round(
                (t_start - tc[0]) / self.p_init[j]), np.round((t_end - tc[0]) / self.p_init[j])
            if truncate:
                mobs_max = np.round((tc[-1] - tc[0]) / self.p_init[j])
                m_max = min(m_max, mobs_max)
            m_all = np.arange(m_min, m_max+1)
            _tcall_linear = t0fit + pfit * m_all
            _idx_in = (t_start < _tcall_linear) & (_tcall_linear < t_end)
            tcall_linear.append(_tcall_linear[_idx_in])
            pidxall = np.r_[pidxall, np.ones_like(
                _tcall_linear[_idx_in])*(j+1)]

        tcall_linear_flatten = np.hstack([t for t in tcall_linear])

        return pidxall, tcall_linear, tcall_linear_flatten

    @partial(jit, static_argnums=(0, 2, 3, 4))
    def get_transit_times_all(self, par_dict, t_start=None, t_end=None, dt=None, transit_orbit_idx=None):
        """compute all model transit times between t_start and t_end

            Args:
                par_dict: dict containing parameters

            Returns:
                tuple: 
                    - 1D flattened array of transit times
                    - fractional energy change

        """
        # set t_start, t_end, dt, time array
        if (t_start is None) or (t_end is None) or (dt is None):
            times, t_start, dt, t_end = self.times, self.t_start, self.dt, self.t_end
        else:
            times, t_start, dt, t_end = jnp.arange(
                t_start, t_end, dt), t_start, dt, t_end

        # compute 1D flattend transit times
        xjac0, vjac0, masses = initialize_jacobi_xv(
            par_dict, t_start)  # initial Jacobi position/velocity
        times, xvjac = integrate_xv(
            xjac0, vjac0, masses, times, nitr=self.nitr_kepler)  # integration
        _orbit_idx, _, tcobs1d = self.tcall_linear(t_start, t_end)
        # idx for orbit, starting from 0
        if transit_orbit_idx is None:
            orbit_idx = _orbit_idx.astype(int) - 1
        else:
            orbit_idx = transit_orbit_idx[_orbit_idx.astype(
                int) - 1].astype(int)
        if self.transit_time_method == 'newton-raphson':
            transit_times = find_transit_times_all(
                orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_transit)
        else:
            transit_times = find_transit_times_kepler_all(
                orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_kepler)
        ediff = get_energy_diff_jac(xvjac, masses, -0.5*dt)

        return transit_times, ediff, _orbit_idx.astype(int) - 1

    def get_transit_times_all_list(self, par_dict, truncate=True, transit_orbit_idx=None):
        """compute all transit times and retunrs a list

            Args:
                par_dict: dict containing parameters
                truncate: if True, only compute transit times up to the last observed time instead of t_end

            Returns:
                list: list of model transit times of length Nplanet; each element is an array of model transit times (length varies for each planet)

        """
        tc_flatten, _, orbit_idx = self.get_transit_times_all(
            par_dict, transit_orbit_idx=transit_orbit_idx)
        tc_list = []
        for j in range(self.nplanet):
            tcj = tc_flatten[orbit_idx == j]
            if truncate:
                tcj = tcj[tcj < self.tcobs[j][-1] + 0.5 * self.p_init[j]]
            tc_list.append(tcj)
        return tc_list

    def check_residuals(self, par_dict=None, tcmodel=None, jitters=None, student=True, normalize_residuals=True, plot=True, fit_mean=False, save=None, xrange=5):
        """check the distribution of residuals, fit them with Student's t distritbution

            Args:
                par_dict: dict containing parameters
                tcmodel: transit time model (1D array)

            Returns:
                tuple:
                    - dictionary: mean of reisudals, SD of residuals
                    - dictionary: parameters of Student's t dist (lndf, lnvar, mean)

        """
        if tcmodel is not None:
            tc = tcmodel
        elif par_dict is not None:
            tc = self.get_transit_times_obs(par_dict)[0]
        else:
            raise ValueError("Either par_dict or tcmodel must be provided.")

        if jitters is not None:
            jitters = np.atleast_1d(jitters)
            if len(jitters) == 1:
                jitters = np.array([jitters[0]] * self.nplanet)
            else:
                assert len(jitters) == self.nplanet
        else:
            jitters = np.zeros(self.nplanet)

        for j in range(self.nplanet):
            fig, ax = plt.subplots(1, 2, figsize=(16, 4))
            idx = self.pidx == j+1
            res = self.tcobs_flatten[idx] - tc[idx]
            err0 = self.errorobs_flatten[idx]
            err = np.sqrt(err0**2 + jitters[j]**2)
            y = np.array(res / err)
            if not normalize_residuals:
                ax[0].errorbar(self.tcobs_flatten[idx], res*1440, yerr=self.errorobs_flatten[idx]*1440, fmt='o', lw=1,
                               label='SD=%.2e' % np.std(res))
                ax[0].set_ylabel('residual (min)')
            else:
                ax[0].errorbar(self.tcobs_flatten[idx], y, yerr=self.errorobs_flatten[idx]/err, fmt='o', lw=1,
                               label='$\chi^2=%.1f$ (%d data)' % (np.sum(y**2), len(y)))
                ax[0].set_ylabel('residual / assigned error')
            ax[0].set_title("planet %d" % (j+1))
            ax[0].set_xlabel("time (days)")
            ax[0].legend(loc='best')

            sd = np.std(y)
            rnds = np.random.randn(int(1e6))
            ax[1].set_yscale("log")
            ax[1].hist(y, histtype='step', lw=1, density=True, color='C0')
            ymin, ymax = ax[1].get_ylim()
            # ax[1].set_ylim(1e-3, ymax*1.5)
            ax[1].set_ylim(ymin/10, ymax*1.5)
            ax[1].set_title("planet %d" % (j+1))
            ax[1].set_xlabel("residual / assigned error")
            ax[1].set_ylabel('frequency (normalized)')
            x0 = np.linspace(-xrange, xrange, 100)
            ax[1].plot(x0, np.exp(-0.5*x0**2/sd**2)/np.sqrt(2*np.pi)/sd, lw=1, color='C0',
                       ls='dashed', label='$\mathrm{SD}=%.2f$ (jitter: %.1e)' % (sd, jitters[j]))
            ax[1].legend(loc='lower right')

        if not student:
            return None

        # Student's t fit
        res = self.tcobs_flatten - tc
        y = np.array(res / self.errorobs_flatten)
        params_st = fit_t_distribution(
            y, plot=plot, fit_mean=fit_mean, save=save, xrange=xrange)

        return {'mean': np.mean(res), 'sd': np.std(res)}, params_st

    def check_timing_precision(self, par_dict, dtfrac=1e-3, nitr_transit=10, nitr_kepler=10):
        """compare get_ttvs output with that computed with a smaller timestep to check the precision

            Args:
                params: JaxTTV parameter array
                dtfrac: (innermost period) * dtfrac is used for the comparison integration

            Returns:
                tuple:
                    - model transit times from get_transit_times_obs
                    - model transit times using a smaller timestep

        """
        tc, de = self.get_transit_times_obs(par_dict)
        print("# fractional energy error (symplectic, dt=%.2e): %.2e" %
              (self.dt, de))

        dtcheck = self.p_init[0] * dtfrac
        assert dtcheck < self.dt, "dtcheck is too large compared to original dt: choose smaller dtfrac."
        self2 = deepcopy(self)
        self2.dt = dtcheck
        self2.times = jnp.arange(self2.t_start, self2.t_end, self2.dt)
        self2.nitr_kepler = nitr_kepler
        self2.nitr_transit = nitr_transit
        tc2, de2 = self2.get_transit_times_obs(par_dict)
        intname = 'symplectic'
        print("# fractional energy error (%s, dt=%.2e): %.2e" %
              (intname, dtcheck, de2))

        tc, tc2 = np.array(tc), np.array(tc2)
        maxdiff = np.max(np.abs(tc-tc2))
        print("# max difference in tc: %.2e days (%.2f sec)" %
              (maxdiff, maxdiff*86400))

        return tc, tc2

    def sample_means_and_stds(self, samples, N=50, truncate=True, original_models=False):
        """compute mean and standard deviation of transit time models from HMC samples

            Args:
                samples: dictionary containing parameter samples (output of mcmc.get_samples())
                N: number of samples to be used for calculation
                truncate: if True, only compute transit times up to the last observed time instead of t_end
                original models: if True, just returns a list of transit-time models

            Returns:
                means and standard deviations of transit time models, list of length(nplanet)

        """
        np.random.seed(123)
        sample_indices = np.random.randint(0, len(samples['period']), N)
        models, means, stds = [], [], []
        for idx in sample_indices:
            pdic_ = {key: val[idx] for key, val in samples.items()}
            tc_list = self.get_transit_times_all_list(pdic_, truncate=truncate)
            models.append(tc_list)

        if original_models:
            return models

        for j in range(len(models[0])):
            models_j = np.array([models[s][j] for s in range(len(models))])
            means.append(np.mean(models_j, axis=0))
            stds.append(np.std(models_j, axis=0))

        return means, stds

    def plot_model(self, tcmodellist, tcobslist=None, errorobslist=None, t0_lin=None, p_lin=None,
                   tcmodelunclist=None, tmargin=None, save=None, marker=None, ylims=None, ylims_residual=None,
                   tcmodellist2=None,
                   unit=1440., lw=1, ylabel='TTV (min)', xlabel='transit time (day)'):
        """plot transit time model

            Args:
                tcmodellist: list of the arrays of model transit times for each planet
                tcobslist: list of the arrays of observed transit times for each planet
                errorobslist: list of the arrays of observed transit time errors for each planet
                t0_lin, p_lin: linear ephemeris used to show TTVs (n_planet,)
                tcmodelunclist: model uncertainty (same format as tcmodellist)
                tcmodellist2 (list of arrays, optional): an optional second set of model transit times to overplot
                tmargin: margin in x axis
                save: if not None, plot is saved as "save_planet#.png"
                marker: marker for model
                unit: TTV unit (defaults to minutes)
                ylabel, xlabel: axis labels in the plots
                ylims, ylims_residual: y ranges in the plots

        """
        if tcobslist is None:
            tcobslist = self.tcobs

        if errorobslist is None:
            if self.errorobs is not None:
                errorobslist = self.errorobs
            else:
                errorobslist = [np.zeros_like(_t) for _t in tcobslist]

        if (t0_lin is None) or (p_lin is None):
            t0_lin, p_lin = self.linear_ephemeris()
            warnings.warn(
                "using t0 and P from a linear fit to the observed transit times.")

        for j, (tcmodel, tcobs, errorobs, t0, p) in enumerate(zip(tcmodellist, tcobslist, errorobslist, t0_lin, p_lin)):
            tcmodel, tcobs, errorobs = np.array(
                tcmodel), np.array(tcobs), np.array(errorobs)

            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                          sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            if tmargin is not None:
                plt.xlim(np.min(tcobs)-tmargin, np.max(tcobs)+tmargin)
            ax.set_ylabel(ylabel)
            ax2.set_xlabel(xlabel)
            tnumobs = np.round((tcobs - t0)/p).astype(int)
            tnummodel = np.round((tcmodel - t0)/p).astype(int)
            ax.errorbar(tcobs, (tcobs-t0-tnumobs*p)*unit, yerr=errorobs*unit, zorder=1000,
                        fmt='o', mfc='white', color='dimgray', label='data', lw=1, markersize=7)
            idxm = tcmodel > 0
            tlin = t0 + tnummodel * p
            ax.plot(tcmodel[idxm], (tcmodel-tlin)[idxm]*unit, '-', marker=marker, lw=lw, mfc='white', color='steelblue',
                    zorder=-1000, label='model', alpha=0.9)

            if tcmodelunclist is not None:
                munc = tcmodelunclist[j]
                ax.fill_between(tcmodel[idxm], (tcmodel-munc-tlin)[idxm]*unit,
                                (tcmodel+munc-tlin)[idxm]*unit,
                                lw=1, color='steelblue', zorder=-1000, alpha=0.2)
            ax.set_title("planet %d" % (j+1))
            if ylims is not None and len(ylims) == len(t0_lin):
                ax2.set_ylim(ylims[j])

            idxm = findidx_map(tcmodel, tcobs)
            ax2.errorbar(tcobs, (tcobs-tcmodel[idxm])*unit, yerr=errorobs*unit, zorder=1000,
                         fmt='o', mfc='white', color='dimgray', label='data', lw=1, markersize=7)
            ax2.axhline(y=0, color='steelblue', alpha=0.6)
            ax2.set_ylabel("residual (min)")
            if ylims_residual is not None and len(ylims_residual) == len(t0_lin):
                ax2.set_ylim(ylims_residual[j])

            if tcmodellist2 is not None:
                tcmodel = tcmodellist2[j]
                tnummodel = np.round((tcmodel - t0)/p).astype(int)
                idxm = tcmodel > 0
                tlin = t0 + tnummodel * p
                ax.plot(tcmodel[idxm], (tcmodel-tlin)[idxm]*unit, '-', marker=marker, lw=lw, mfc='white', color='C1',
                        zorder=-1000, label='model2', alpha=0.9)

            # change legend order
            handles, labels = ax.get_legend_handles_labels()
            order = [1, 0]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                      loc='upper left', bbox_to_anchor=(1, 1))

            fig.tight_layout(pad=0.05)

            if save is not None:
                plt.savefig(save+"_planet%d.png" %
                            (j+1), dpi=200, bbox_inches="tight")


@partial(jit, static_argnums=(5,))
def integrate_orbits_symplectic(xjac0, vjac0, masses, times, dt, nitr_kepler):
    """symplectic integration of the orbits

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
    xjac, vjac = kepler_step_map(
        xvjac[:, 0, :, :], xvjac[:, 1, :, :], masses, dt_correct)

    # conversion to CM frame
    xcm, vcm, acm = xvjac_to_xvacm(xjac, vjac, masses)

    return t, xcm, vcm, acm, de_frac


@jit
def integrate_orbits_hermite(xjac0, vjac0, masses, times):
    """Hermite integration of the orbits

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
    xcm, vcm, acm = xvacm[:, 0, :, :], xvacm[:, 1, :, :], xvacm[:, 2, :, :]
    de_frac = get_energy_diff(xvacm, masses)
    return t, xcm, vcm, acm, de_frac
