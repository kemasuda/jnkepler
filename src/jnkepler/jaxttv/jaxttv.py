__all__ = ["JaxTTV", "elements_to_pdic", "params_to_elements"]

#%%
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
from .utils import get_ediff, get_energy_vmap, initialize_from_elements, xv_to_elements, BIG_G
from .hermite4 import integrate_elements, find_transit_times_planets, find_transit_times_nodata, findidx_map
from .symplectic import get_ttvs as jttvfast
from jax import jit, grad

#%%
from jax.config import config
config.update('jax_enable_x64', True)

#%%
M_earth = 3.0034893e-6
def elements_to_pdic(elements, masses, outkeys=None, force_coplanar=True):
    npl = len(masses) - 1
    pdic = {}
    pdic['pmass'] = masses[1:] / M_earth
    pdic['period'] = jnp.array([elements[j][0] for j in range(npl)])
    pdic['ecosw'] = jnp.array([elements[j][1] for j in range(npl)])
    pdic['esinw'] = jnp.array([elements[j][2] for j in range(npl)])
    if force_coplanar:
        copl = 0.
    else:
        copl = 1.
    pdic['cosi'] = jnp.array([elements[j][3]*copl for j in range(npl)])
    pdic['lnode'] = jnp.array([elements[j][4]*copl for j in range(npl)])
    pdic['tic'] = jnp.array([elements[j][5] for j in range(npl)])
    pdic['ecc'] = jnp.sqrt(pdic['ecosw']**2 + pdic['esinw']**2)
    pdic['omega'] = jnp.arctan2(pdic['esinw'], pdic['ecosw'])
    pdic['lnmass'] = jnp.log(masses[1:])
    pdic['mass'] = masses[1:]

    pdic['ecc'] = jnp.sqrt(pdic['ecosw']**2 + pdic['esinw']**2)
    pdic['cosw'] = pdic['ecosw'] / jnp.fmax(pdic['ecc'], 1e-2)
    pdic['sinw'] = pdic['esinw'] / jnp.fmax(pdic['ecc'], 1e-2)

    if outkeys is None:
        return pdic

    for key in list(pdic.keys()):
        if key not in outkeys:
            pdic.pop(key)
    return pdic

def params_to_elements(params, npl):
    elements = jnp.array(params[:-npl].reshape(npl, -1))
    masses = jnp.exp(jnp.hstack([[0], params[-npl:]]))
    return elements, masses

#%%
class JaxTTV:
    def __init__(self, t_start, t_end, dt, symplectic=True):
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.times = jnp.arange(t_start, t_end, dt)
        self.symplectic = symplectic
        if symplectic:
            print ("# sympletic integrator is used.")
        else:
            print ("# hermite integrator is used.")

    def set_tcobs(self, tcobs, p_init, errorobs=None):
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

        print ("# integration starts at:".ljust(35) + "%.2f"%self.t_start)
        print ("# first transit time in data:".ljust(35) + "%.2f"%np.min(self.tcobs_flatten))
        print ("# last transit time in data:".ljust(35) + "%.2f"%np.max(self.tcobs_flatten))
        print ("# integration ends at:".ljust(35) + "%.2f"%self.t_end)
        print ("# integration time step:".ljust(35) + "%.4f (1/%d of innermost period)"%(self.dt, np.nanmin(p_init)/self.dt))

    def update_period(self):
        p_new = []
        for j in range(self.nplanet):
            tc = self.tcobs[j]
            m = np.round((tc - tc[0]) / self.p_init[j])
            pfit, t0fit = np.polyfit(m, tc, deg=1)
            p_new.append(pfit)
        return np.array(p_new)

    def integrate(self, elements, masses, t_start, t_end, dt):
        times = jnp.arange(t_start, t_end, dt)
        t, xva = integrate_elements(elements, masses, times, t_start)
        energy = get_energy_vmap(xva[:,0,:,:], xva[:,1,:,:], masses)
        return t, xva, energy

    @partial(jit, static_argnums=(0,))
    def get_ttvs(self, elements, masses):
        if self.symplectic:
            return jttvfast(self, elements, masses)
        else:
            t, xva = integrate_elements(elements, masses, self.times, self.t_start)
            de_frac = get_ediff(xva, masses)
            return find_transit_times_planets(t, xva, self.tcobs, masses), de_frac

    def get_ttvs_nojit(self, elements, masses, t_start=None, t_end=None, dt=None, flatten=False, nitr=5):
        if t_start is not None:
            times, t0 = jnp.arange(t_start, t_end, dt), t_start
        else:
            times, t0 = self.times, self.t_start
        t, xva = integrate_elements(elements, masses, times, t0)
        de_frac = get_ediff(xva, masses)

        tcarr = []
        for pidx in range(1, len(masses)):
            tc = find_transit_times_nodata(t, xva[:,0,:,:], xva[:,1,:,:], xva[:,2,:,:], pidx, masses, nitr=nitr)
            #tc = find_transit_times_nodata(t, xva, pidx, masses)
            tcarr.append(tc)
        if flatten:
            tcarr = np.hstack(tcarr)

        return tcarr, de_frac

    def quicklook(self, model, sigma=None, save=None):
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

    def check_prec(self, params, dtfrac=1e-3, nitr=10):
        tc, de = self.get_ttvs(*params_to_elements(params, self.nplanet))
        print ("# fractional energy error (symplectic, dt=%.2e): %.2e" % (self.dt,de))

        elements, masses = params_to_elements(params, self.nplanet)
        dtcheck = self.p_init[0] * dtfrac
        tc2, de2 = self.get_ttvs_nojit(elements, masses, t_start=self.t_start, t_end=self.t_end, dt=dtcheck,
                                       flatten=True, nitr=nitr)
        tc2 = tc2[np.array(findidx_map(tc2, tc))]
        print ("# fractional energy error (Hermite, dt=%.2e): %.2e" % (dtcheck, de2))
        maxdiff = np.max(np.abs(tc-tc2))
        print ("# max difference in tc: %.2e days (%.2f sec)"%(maxdiff, maxdiff*86400))

        return tc

    def optim(self, dp=5e-1, dtic=1e-1, emax=0.5, mmin=1e-7, mmax=1e-3, cosilim=[-1e-6,1e-6], olim=[-1e-6,1e-6],
          amoeba=False, plot=True, save=None, pinit=None, jacrev=False):
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

    def get_elements(self, params, WHsplit=False):
        elements, masses = params_to_elements(params, self.nplanet)
        xjac, vjac = initialize_from_elements(elements, masses, self.t_start)

        if WHsplit: # for H_Kepler defined in WH splitting (i.e. TTVFast)
            ki = BIG_G * masses[0] * jnp.cumsum(masses)[1:] / jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])
        else: # total interior mass
            ki = BIG_G * jnp.cumsum(masses)[1:]

        return xv_to_elements(xjac, vjac, ki)
