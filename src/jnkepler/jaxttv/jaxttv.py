__all__ = ["jaxttv", "elements_to_pdic", "params_to_elements"]

#%%
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
from .utils import get_ediff, get_energy_vmap, initialize_from_elements, xv_to_elements, BIG_G
from .hermite4 import integrate_elements, find_transit_times_planets, find_transit_times_nodata
from .symplectic import get_ttvs as jttvfast#, test
from jax import jit, grad
#import jax, jaxlib, numpyro
#jax.__version__
#jaxlib.__version__
#numpyro.__version__

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
class jaxttv:
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
            if not len(tc):
                continue

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

    """
    @partial(jit, static_argnums=(0,))
    def get_ttvs(self, elements, masses):
        return jttvfast(self, elements, masses)

    @partial(jit, static_argnums=(0,))
    def get_ttvs_hermite4(self, elements, masses):
        t, xva = integrate_elements(elements, masses, self.times, self.t_start)
        de_frac = get_ediff(xva, masses)
        return find_transit_times_planets(t, xva, self.tcobs, masses), de_frac
    """

    def get_ttvs_nojit(self, elements, masses, t_start=None, t_end=None, dt=None, flatten=False):
        if t_start is not None:
            times, t0 = jnp.arange(t_start, t_end, dt), t_start
        else:
            times, t0 = self.times, self.t_start
        t, xva = integrate_elements(elements, masses, times, t0)
        de_frac = get_ediff(xva, masses)

        tcarr = []
        for pidx in range(1, len(masses)):
            tc = find_transit_times_nodata(t, xva[:,0,:,:], xva[:,1,:,:], xva[:,2,:,:], pidx, masses)
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

    def optim(self, dp=5e-1, dtic=1e-1, emax=0.5, mmin=1e-7, mmax=1e-3, cosilim=[-1e-6,1e-6], olim=[-1e-6,1e-6], amoeba=False, plot=True, save=None, noscale=True, pinit=None):
        from scipy.optimize import curve_fit
        import time

        npl = self.nplanet

        params_lower, params_upper = [], []
        pnames, scales = [], []
        for j in range(npl): # need to be changed to take into account non-transiting planets
            params_lower += [self.p_init[j]-dp, -emax, -emax, cosilim[0], olim[0], self.tcobs[j][0]-dtic]
            params_upper += [self.p_init[j]+dp, emax+1e-2, emax+1e-2, cosilim[1], olim[1], self.tcobs[j][0]+dtic]
            pnames += ["p%d"%(j+1), "ec%d"%(j+1), "es%d"%(j+1), "cosi%d"%(j+1), "om%d"%(j+1), "tic%d"%(j+1)]
            scales += [self.p_init[j], 1, 1, 1, jnp.pi, self.tcobs[j][0]]
        params_lower += [jnp.log(mmin)] * npl
        params_upper += [jnp.log(mmax)] * npl
        pnames += ["m%d"%(j+1) for j in range(npl)]
        scales = jnp.array((scales) + [10.]*npl).ravel()
        params_lower = jnp.array(params_lower).ravel()
        params_upper = jnp.array(params_upper).ravel()
        if noscale:
            scales = 1.

        lower_bounds = params_lower / scales
        upper_bounds = params_upper / scales
        bounds = (lower_bounds, upper_bounds)
        if pinit is None:
            pinit = 0.5 * (lower_bounds + upper_bounds)

        def getmodel(params):
            params *= scales
            elements, masses = params_to_elements(params, npl)
            model = self.get_ttvs(elements, masses)[0]
            return model

        #objective = lambda params: jnp.sum(myfunct(params)[1]**2)
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
        print ("running LM optimization...")
        popt, pcov = curve_fit(func, None, self.tcobs_flatten, p0=pinit, sigma=self.errorobs_flatten, bounds=bounds)
        print ("objective function: %.2f (%d data)"%(objective(popt), len(self.tcobs_flatten)))
        print ("# elapsed time (least square): %.1f sec" % (time.time()-start_time))
        pfinal = popt

        if plot:
            self.quicklook(getmodel(pfinal), save=save)

        return pfinal

    def optimal_params(self, dp=5e-1, dtic=1e-1, emax=0.5, mmin=1e-7, mmax=1e-3, cosilim=[0,0], olim=[0,0], amoeba=False, plot=True, save=None, noscale=True):
        import jaxopt
        from pympfit import lmfit

        npl = self.nplanet

        params_lower, params_upper = [], []
        pnames, scales = [], []
        for j in range(npl): # need to be changed to take into account non-transiting planets
            params_lower += [self.p_init[j]-dp, -emax, -emax, cosilim[0], olim[0], self.tcobs[j][0]-dtic]
            params_upper += [self.p_init[j]+dp, emax+1e-2, emax+1e-2, cosilim[1], olim[1], self.tcobs[j][0]+dtic]
            pnames += ["p%d"%(j+1), "ec%d"%(j+1), "es%d"%(j+1), "cosi%d"%(j+1), "om%d"%(j+1), "tic%d"%(j+1)]
            scales += [self.p_init[j], 1, 1, 1, jnp.pi, self.tcobs[j][0]]
        params_lower += [jnp.log(mmin)] * npl
        params_upper += [jnp.log(mmax)] * npl
        pnames += ["m%d"%(j+1) for j in range(npl)]
        scales = jnp.array((scales) + [10.]*npl).ravel()
        params_lower = jnp.array(params_lower).ravel()
        params_upper = jnp.array(params_upper).ravel()
        if noscale:
            scales = 1.

        lower_bounds = params_lower / scales
        upper_bounds = params_upper / scales
        bounds = (lower_bounds, upper_bounds)
        pinit = 0.5 * (lower_bounds + upper_bounds)

        def getmodel(params):
            params *= scales
            elements, masses = params_to_elements(params, npl)
            model = self.get_ttvs(elements, masses)[0]
            return model

        def myfunct(params, fjac=None, parname=None):
            model = np.array(getmodel(params))
            res = (self.tcobs_flatten - model) / self.errorobs_flatten
            status = 0
            return [status, res]

        #objective = lambda params: jnp.sum(myfunct(params)[1]**2)
        objective = lambda params: jnp.sum(((self.tcobs_flatten - getmodel(params)) / self.errorobs_flatten)**2)
        print ("initial objective function: %.2f (%d data)"%(objective(pinit), len(self.tcobs_flatten)))

        if amoeba:
            from scipy.optimize import minimize
            print ()
            print ("running Nelder-Mead optimization...")
            res = minimize(objective, pinit, bounds=np.array(bounds).T, method='Nelder-Mead', options={'adaptive': False})
            print ("objective function: %.2f (%d data)"%(objective(res.x), len(self.tcobs_flatten)))
            pinit = res.x

        try:
            print ()
            print ("running LM optimization...")
            sol = lmfit(myfunct, bounds, pnames, initial=pinit, out=None)
            print ("objective function: %.2f (%d data)"%(objective(sol.params), len(self.tcobs_flatten)))
            pfinal = sol.params

        except:
            print ("mpfit failed.")
            print ()
            print ("running L-BFGS-B optimization...")
            solver = jaxopt.ScipyBoundedMinimize(fun=objective, method="l-bfgs-b")
            sol = solver.run(pinit, bounds=bounds)
            #"""
            """
            from scipy.optimize import minimize
            res = minimize(objective, pinit, bounds=np.array(bounds).T, method='L-BFGS-B')
            print ("objective function: %.2f (%d data)"%(objective(res.x), len(self.tcobs_flatten)))
            pfinal = res.x
            """

        if plot:
            self.quicklook(getmodel(sol.params), save=save)

        return sol.params

    def get_elements(self, params, time, wh=False, dt=None):
        elements, masses = params_to_elements(params, self.nplanet)
        if dt is None:
            dt = self.dt
        if time < self.t_start + self.dt:
            tprint = self.t_start
            xjac, vjac = initialize_from_elements(elements, masses, self.t_start)
        else:
            t, xva, _ = self.integrate(elements, masses, self.t_start, time, dt)
            tprint = t[-1]
            x, v, a = xva[-1]
            xjac, vjac = (x[:,:] - x[0,:])[1:], (v[:,:] - v[0,:])[1:]
        if wh: # for H_Kepler defined in WH splitting
            ki = BIG_G * masses[0] * jnp.cumsum(masses)[1:] / jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])
        else:
            ki = BIG_G * jnp.cumsum(masses)[1:]
        print ("# elements at time %f (time specified: %f)"%(tprint,time))
        return xv_to_elements(xjac, vjac, ki)

#%%
"""
elements = jnp.array([[365.25, 0.4*jnp.cos(0.1*jnp.pi), 0.4*jnp.sin(0.1*jnp.pi), 0, 0.1*jnp.pi, 40], [365.25*2., 0., 0, 0, 0.1*jnp.pi, 40]])
masses = jnp.array([1, 300e-6, 300e-6])

#%%
elements = jnp.array([[10, 0.1*jnp.cos(0.1*jnp.pi), 0.1*jnp.sin(0.1*jnp.pi), 0, 0, 2], [20.2, 0.2*jnp.cos(0.1*jnp.pi), 0.2*jnp.sin(0.1*jnp.pi), 0, 0, 3]])
masses = jnp.array([1, 30e-6, 30e-6])

#%%
elements = jnp.array([[10, 0.1*jnp.cos(0.1*jnp.pi), 0.1*jnp.sin(0.1*jnp.pi), 0, 0, 2], [20.2, 0.2*jnp.cos(0.1*jnp.pi), 0.2*jnp.sin(0.1*jnp.pi), 0, 0, 3], [30, 0.1*jnp.cos(-0.5*jnp.pi), 0.1*jnp.sin(-0.5*jnp.pi), 0, 0, 4]])
masses = jnp.array([1, 10e-6, 10e-6, 10e-6])

#%%
dt = 0.2 * 2
t_start, t_end = 0, 4 * 365.25
jttv = jaxttv(t_start, t_end, dt)

#%%
#t, xva, etot = jttv.integrate(elements, masses, t_start, t_end, dt)
#plt.plot(t, etot/etot[0]-1., '-')

#%%
tcobs, _ = jttv.get_ttvs_nojit(elements, masses, t_start, t_end, 0.001)

#%%
p_init = [elements[j][0] for j in range(len(masses)-1)]
jttv.set_tcobs(tcobs, p_init)

#%%
%timeit tpars, de = jttv.get_ttvs(elements, masses)
tpars, de = jttv.get_ttvs(elements, masses)
print ("fractional energy change:", de)

#%%
jttv.quicklook(tpars)#, save='test')

#%%
dtc = tpars - jnp.hstack(jttv.tcobs)
plt.xlabel("transit time difference (sec)")
plt.hist(np.array(dtc)*86400)

#%%
# 6331, 21699 for macbookpro, no change with nitr
# 4949, 8352 w/o tc iteration
# 6320, 21674 without final for
#, 15659 w/o final for
# 5800, 17950 for dE iteration
# 6026, 18418 for symp2
# 4966, 12749 for symp2 w/o jit
# imac: 3967, 9673 for dE iteration
# 2975, 7098 for dE, w/o mapTO
# 2776, 6576 for dE, w/o mapTO, w/o jit
# updated imac: 3764, 9403 for dE, w/o mapTO, w/o jit
# updated imac: 3976, 10121 for symp2
# 4249, 10167 for symp2, w/ mapTO
# 3964, 9507 for symp2, w/ mapTO, w/o jit
# 3102, 7224 for symp2, w/o mapTO, w/o jit
f = lambda elements, masses: jnp.sum(jttv.get_ttvs(elements, masses)[0])
#f = lambda elements, masses: jnp.sum(test(jttv, elements, masses)[0])
gf = jit(grad(f))
#%timeit ttvf(elements, masses)
#gttvf(elements, masses)
#%timeit gttvf(elements, masses)

#%%
f(elements, masses)
gf(elements, masses)

#%%
# 3935, 10363 for macbookpro
# 3397, 9620 with inaccurate tc
# imac: 2500, 5820
#jttv.quicklook(jttv.get_ttvs_hermite4(elements, masses)[0])
f = lambda elements, masses: jnp.sum(jttv.get_ttvs_hermite4(elements, masses)[0])
gf = jit(grad(f))
f(elements, masses)
gf(elements, masses)

#%%
from jax import make_jaxpr
make_jaxpr(f)(elements, masses)
make_jaxpr(gf)(elements, masses)
"""
