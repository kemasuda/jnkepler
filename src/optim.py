#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner
from jaxttv import jaxttv
import jaxopt
from jax.config import config
config.update('jax_enable_x64', True)

#%%
"""
import seaborn as sns
sns.set(style='ticks', font_scale=1.6, font='times')
from matplotlib import rc
rc('text', usetex=True*0)
plt.rcParams['savefig.dpi']=200
"""

M_earth = 3.0034893e-6

#%%
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax.random as random

def params_dic(elements, masses):
    npl = len(masses) - 1
    pdic = {}
    pdic['pmass'] = masses[1:] / M_earth
    pdic['period'] = np.array([elements[j][0] for j in range(npl)])
    pdic['ecosw'] = np.array([elements[j][1] for j in range(npl)])
    pdic['esinw'] = np.array([elements[j][2] for j in range(npl)])
    pdic['cosi'] = np.array([elements[j][3] for j in range(npl)])
    pdic['lnode'] = np.array([elements[j][4] for j in range(npl)])
    pdic['tic'] = np.array([elements[j][5] for j in range(npl)])
    pdic['ecc'] = np.sqrt(pdic['ecosw']**2 + pdic['esinw']**2)
    pdic['omega'] = np.arctan2(pdic['esinw'], pdic['ecosw'])
    pdic['lnmass'] = np.log(masses[1:])
    return pdic

#%%
dt = 0.1 * 0.1
t_start, t_end = 0, 365.25 * 0.5
jttv = jaxttv(t_start, t_end, dt)

#%% p, e, i, omega, lnode, tc
elements = jnp.array([[10, 0.1, jnp.pi*0.5, 0.1*jnp.pi, 0, 2], [20.2, 0.2, jnp.pi*0.5, 0.1*jnp.pi, 0, 3]])
elements = jnp.array([[10, 0.1*jnp.cos(0.1*jnp.pi), 0.1*jnp.sin(0.1*jnp.pi), 0, 0, 2], [20.2, 0.2*jnp.cos(0.1*jnp.pi), 0.2*jnp.sin(0.1*jnp.pi), 0, 0, 3]])
masses = jnp.array([1, 30e-6, 30e-6])
pdic_truth = params_dic(elements, masses)
pdic_truth

#%%
tcobs = jttv.get_ttvs_nodata(elements, masses, t_start, t_end, dt)
tc1, tc2 = tcobs
tcerr = 1e-3
np.random.seed(123)
tc1 += np.random.randn(len(tc1)) * tcerr
tc2 += np.random.randn(len(tc2)) * tcerr
tc1err = jnp.ones_like(tc1) * tcerr
tc2err = jnp.ones_like(tc2) * tcerr
data = jnp.hstack([tc1, tc2])
error = jnp.hstack([tc1err, tc2err])

#%%
p_init = [elements[0][0], elements[1][0]]
jttv.set_tcobs([tc1, tc2], p_init, errorobs=[tc1err, tc2err])

#%%
jttv.quicklook(jttv.get_ttvs(elements, masses))

#%%
def optimal_params(dp=1e-2, dtic=1e-2, emax=0.2, mmin=1e-7, mmax=1e-3, plot=True):
    scales = 1.
    npl = jttv.nplanet

    def getmodel(params):
        params *= scales
        elements = jnp.array(params[:-npl].reshape(npl,-1))
        masses = jnp.exp(jnp.hstack([[0], params[-npl:]]))
        model = jttv.get_ttvs(elements, masses)
        return model

    def objective(params):
        model = getmodel(params)
        res = (jttv.tcobs_flatten - model) / jttv.errorobs_flatten
        return 0.5 * jnp.dot(res, res)

    params_lower, params_upper = [], []
    cosimin, cosimax = 0, 0
    omin, omax = 0, 0
    for j in range(npl):
        params_lower += [p_init[j]-dp, -emax, -emax, cosimin, omin, jttv.tcobs[j][0]-dtic]
        params_upper += [p_init[j]+dp, emax, emax, cosimax, omax, jttv.tcobs[j][0]+dtic]
    params_lower += [jnp.log(mmin)] * npl
    params_upper += [jnp.log(mmax)] * npl
    params_lower = jnp.array(params_lower).ravel()
    params_upper = jnp.array(params_upper).ravel()
    lower_bounds = params_lower / scales
    upper_bounds = params_upper / scales
    bounds = (lower_bounds, upper_bounds)
    pinit = 0.5 * (lower_bounds + upper_bounds)

    print ("inital objective function: %.2f (%d data)"%(objective(pinit), len(jttv.tcobs_flatten)))

    solver1 = jaxopt.ScipyBoundedMinimize(fun=objective, method="l-bfgs-b")
    solver2 = jaxopt.ScipyBoundedMinimize(fun=objective, method="Nelder-Mead", tol=1e-3)

    print ()
    print ("running Nelder-Mead optimization...")
    sol = solver2.run(pinit, bounds=bounds)
    print ("objective function: %.2f (%d data)"%(objective(sol.params), len(jttv.tcobs_flatten)))

    print ()
    print ("running L-BFGS-B optimization...")
    for i in range(2):
        sol = solver1.run(sol.params, bounds=bounds)
    print ("objective function: %.2f (%d data)"%(objective(sol.params), len(jttv.tcobs_flatten)))

    if plot:
        jttv.quicklook(getmodel(sol.params))

    return sol.params

#%%
params_best = optimal_params()

#%%
def initdict(params, pnames=['period', 'ecosw', 'esinw', 'cosi', 'lnode', 'tic', 'mass']):
    pdic = {}
    for i,p in enumerate(pnames):
        if p!='mass':
            arr = []
            for j in range(jttv.nplanet):
                arr.append(params[i+j*6])
            pdic[p] = jnp.array(arr)
        else:
            pdic[p] = jnp.exp(jnp.hstack([[0], params[-jttv.nplanet:]]))
    pdic['ecc'] = jnp.sqrt(pdic['ecosw']**2 + pdic['esinw']**2)
    #pdic['omega'] = jnp.arctan2(pdic['esinw'], pdic['ecosw'])
    pdic['cosw'] = pdic['ecosw'] / pdic['ecc']
    pdic['sinw'] = pdic['esinw'] / pdic['ecc']
    pdic['lnmass'] = jnp.log(pdic['mass'][1:])
    pdic.pop('ecosw')
    pdic.pop('esinw')
    pdic.pop('mass')
    return pdic

#%%
pdic_init = initdict(params_best)
pdic_init

#%%
tic_guess = jnp.array([2, 3])
p_guess = jnp.array(p_init)
from numpyro.infer import init_to_value
def model():
    tic = numpyro.sample("tic", dist.Uniform(low=tic_guess-1e-1, high=tic_guess+1e-1))
    period = numpyro.sample("period", dist.Uniform(low=p_guess-1e-1, high=p_guess+1e-1))
    ecc = numpyro.sample("ecc", dist.Uniform(low=jnp.array([0, 0]), high=jnp.array([0.3, 0.3])))

    cosw = numpyro.sample("cosw", dist.Normal(scale=jnp.array([1, 1])))
    sinw = numpyro.sample("sinw", dist.Normal(scale=jnp.array([1, 1])))
    omega = jnp.arctan2(sinw, cosw)
    numpyro.deterministic("omega", omega)

    #cosO = numpyro.sample("cosO", dist.Normal(scale=jnp.array([1, 1])))
    #sinO = numpyro.sample("sinO", dist.Normal(scale=jnp.array([1, 1])))
    #lnode = jnp.arctan2(sinO, cosO)
    lnode = jnp.array([0, 0])
    numpyro.deterministic("lnode", lnode)
    cosi = jnp.array([0., 0.])
    numpyro.deterministic("cosi", cosi)
    lnmass = numpyro.sample("lnmass", dist.Uniform(low=jnp.array([-16, -16]), high=jnp.array([-6, -6])))
    #mass = jnp.hstack([1, jnp.exp(lnmass)])
    mass = jnp.exp(lnmass)
    numpyro.deterministic("mass", mass)

    elements = jnp.array([period, ecc*cosw, ecc*sinw, cosi, lnode, tic]).T
    numpyro.deterministic("elements", elements)

    tcmodel = jttv.get_ttvs(elements, jnp.hstack([1, mass]))
    #tcmodel = jnp.where(tcmodel==tcmodel, tcmodel, 0)
    numpyro.deterministic("tcmodel", tcmodel)
    numpyro.sample("obs", dist.Normal(loc=tcmodel, scale=error), obs=data)

#%%
init_strategy = init_to_value(values=pdic_init)

#%%
kernel = numpyro.infer.NUTS(model, target_accept_prob=0.8, init_strategy=init_strategy)

#%%
nw, ns = 100, 100
nw, ns = 500, 1000 # 16hr

#%%
mcmc = numpyro.infer.MCMC(kernel, num_warmup=nw, num_samples=ns)

#%%
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, extra_fields=('potential_energy',))

#%%
mcmc.print_summary()

#%%
import dill
with open("mcmc2.pkl", "wb") as f:
    dill.dump(mcmc, f)

#%%
samples = mcmc.get_samples()

#%%
mmodel, smodel = jnp.mean(samples['tcmodel'], axis=0), jnp.std(samples['tcmodel'], axis=0)
jttv.quicklook(mmodel, sigma=smodel)

#%%
samples['pmass'] = samples['mass'] / M_earth
keys = ['pmass', 'tic', 'period', 'ecc', 'omega']
keys = ['lnmass', 'tic', 'period', 'ecc', 'omega']

#%%
for j in range(jttv.nplanet):
    hyper = pd.DataFrame(data=dict(zip(keys, [samples[k][:,j] for k in keys])))
    fig = corner.corner(hyper, labels=keys, show_titles="%.2f", truths=[pdic_truth[k][j] for k in keys])
    plt.savefig("corner%d.png"%(j+1), dpi=200, bbox_inches="tight")


#%%
"""
#p_new = jttv.update_period()
npl = jttv.nplanet

#%%
params_lower, params_upper = [], []
dp, dtic = 1e-2, 1e-1
emax = 0.5
cosimin, cosimax = 0, 0
omin, omax = 0, 0
mmin, mmax = jnp.log(1e-7), jnp.log(1e-3)
for j in range(jttv.nplanet):
    params_lower += [p_init[j]-dp, -emax, -emax, cosimin, omin, jttv.tcobs[j][0]-dtic]
    params_upper += [p_init[j]+dp, emax, emax, cosimax, omax, jttv.tcobs[j][0]+dtic]
params_lower += [mmin]*jttv.nplanet
params_upper += [mmax]*jttv.nplanet
params_lower = jnp.array(params_lower).ravel()
params_upper = jnp.array(params_upper).ravel()

#%%
params_init = jnp.hstack([jnp.hstack([elements[j] for j in range(npl)]), jnp.log(masses[1:])])
#params_lower = jnp.array([10-1e-3, 0, 0.5*jnp.pi, 0, 0, 2-1e-2, 20.2-1e-3, 0, 0.5*jnp.pi, 0, 0, 3-1e-2, 0, 0])
#params_upper = jnp.array([10+1e-3, 0.3, 0.5*jnp.pi, 1, 0, 2+1e-2, 20.2+1e-3, 0.3, 0.5*jnp.pi, 1, 0, 3+1e-2, 100e-6, 100e-6])
#scales = jnp.hstack([jnp.tile(jnp.array([10, 1, jnp.pi, jnp.pi, jnp.pi, 2]), 2), jnp.ones(npl)*3e-5])
scales = 1
#scales = jnp.hstack([jnp.tile(jnp.array([10, 1, 1, 1, jnp.pi, 2]), 2), jnp.ones(npl)*3e-5])

def getmodel(params):
    params *= scales
    elements = jnp.array(params[:-npl].reshape(npl,-1))
    masses = jnp.exp(jnp.hstack([[0], params[-npl:]]))
    model = jttv.get_ttvs(elements, masses)
    return model

def objective(params):
    model = getmodel(params)
    res = (data - model) / error
    return jnp.dot(res, res)

#%%
print (objective(params_init/scales))

#%%
jttv.quicklook(getmodel(params_init/scales))

#%%
lower_bounds = params_lower / scales
upper_bounds = params_upper / scales
bounds = (lower_bounds, upper_bounds)
pinit = 0.5 * (lower_bounds + upper_bounds)

#%%
print (objective(pinit))
jttv.quicklook(getmodel(pinit))

#%%
from jaxopt.projection import projection_box
solver1 = jaxopt.ScipyBoundedMinimize(fun=objective, method="l-bfgs-b")
solver2 = jaxopt.ScipyBoundedMinimize(fun=objective, method="Nelder-Mead")
powell = jaxopt.ScipyBoundedMinimize(fun=objective, method="Powell")
tnc = jaxopt.ScipyBoundedMinimize(fun=objective, method="TNC")
#pg = jaxopt.ProjectedGradient(fun=objective, projection=projection_box)

#%%
#pg_sol = pg.run(pinit, hyperparams_proj=bounds)
#print (objective(pg_sol.params))

#%%
#sol = powell.run(pinit, bounds=bounds)
sol = solver2.run(pinit, bounds=bounds)
for i in range(2):
    sol = solver1.run(sol.params, bounds=bounds)

#%%
print (objective(sol.params))

#%%
jttv.quicklook(getmodel(sol.params))

#%%
sol.params * scales

#%%
params_init

#%%
def model(p):
    params = p * scales
    elements = jnp.array(params[:-npl].reshape(npl,-1))
    masses = jnp.hstack([[1], params[-npl:]])
    return jttv.get_ttvs(elements, masses)

def objective(p):
    res = (data - model(p)) / error
    return jnp.dot(res, res)

#%%
lbfgsb = jaxopt.ScipyBoundedMinimize(fun=objective, method="l-bfgs-b")
#pinit = params_init / scales
#pinit *= np.random.rand(len(pinit)) * 1e-2
lower_bounds = params_lower / scales
upper_bounds = params_upper / scales
bounds = (lower_bounds, upper_bounds)
pinit = 0.5 * (lower_bounds + upper_bounds)

#%%
lbfgsb_sol = lbfgsb.run(pinit, bounds=bounds)

#%%
plt.errorbar(tc1[1:], jnp.diff(tc1), fmt='o', yerr=1e-3)
plt.plot(tc1[1:], jnp.diff(model(lbfgsb_sol.params)[:len(tc1)]), '.-')

#%%
plt.errorbar(tc2[1:], jnp.diff(tc2), fmt='o', yerr=1e-3)
plt.plot(tc2[1:], jnp.diff(model(lbfgsb_sol.params)[len(tc1):]), '.-')

#%%
params_init
pinit * scales
lbfgsb_sol.params * scales

#%%
model(params_init/scales, scales)
objective(params_init/scales)

#%%
gd = jaxopt.GradientDescent(fun=objective, maxiter=10, stepsize=1.e-6)
res = gd.run(init_params=params_init/scales)

#%%
res[0] * scales
model(res[0], scales)

#%%
plt.plot(tc1[1:], np.diff(tc1)-np.mean(np.diff(tc1)), '.')
plt.plot(tc2[1:], np.diff(tc2)-np.mean(np.diff(tc2)), '.')
plt.plot(_tc1[1:], np.diff(_tc1)-np.mean(np.diff(_tc1)), '-')
plt.plot(_tc2[1:], np.diff(_tc2)-np.mean(np.diff(_tc2)), '-')

#%%
p_guess = jnp.array([np.mean(np.diff(tc1)), np.mean(np.diff(tc2))])
tic_guess = jnp.array([tc1[0], tc2[0]])
tic_guess

#%%
def model(data, error):
    #tic = numpyro.sample("tic", dist.Normal(loc=jnp.array([t1_guess, t2_guess]), scale=jnp.array([1e-2, 1e-2])))
    #period = numpyro.sample("period", dist.Normal(loc=jnp.array([p1_guess, p2_guess]), scale=jnp.array([1e-3, 1e-3])))
    #tic = jnp.array([t1_guess, t2_guess])
    #period = jnp.array([p1_guess, p2_guess])
    tic = numpyro.sample("tic", dist.Uniform(low=tic_guess-1e-1, high=tic_guess+1e-1))
    period = numpyro.sample("period", dist.Uniform(low=p_guess-1e-3, high=p_guess+1e-3))
    ecc = numpyro.sample("ecc", dist.Uniform(low=jnp.array([0, 0]), high=jnp.array([0.3, 0.3])))

    cosw = numpyro.sample("cosw", dist.Normal(scale=jnp.array([1, 1])))
    sinw = numpyro.sample("sinw", dist.Normal(scale=jnp.array([1, 1])))
    omega = jnp.arctan2(sinw, cosw)
    numpyro.deterministic("omega", omega)
    #cosO = numpyro.sample("cosO", dist.Normal(scale=jnp.array([1, 1])))
    #sinO = numpyro.sample("sinO", dist.Normal(scale=jnp.array([1, 1])))
    #lnode = jnp.arctan2(sinO, cosO)
    lnode = jnp.array([0, 0])
    numpyro.deterministic("lnode", lnode)
    inc = jnp.array([0.5*jnp.pi, 0.5*jnp.pi])
    lnmass = numpyro.sample("lnmass", dist.Uniform(low=jnp.array([-16, -16]), high=jnp.array([-8, -8])))
    mass = jnp.hstack([1, jnp.exp(lnmass)])
    numpyro.deterministic("mass", mass)

    elements = jnp.array([period, ecc, inc, omega, lnode, tic]).T
    numpyro.deterministic("elements", elements)

    tc1model, tc2model = jttv.get_ttvs(elements, mass)
    #tc1model, tc2model = jttv.get_ttvs(elements, masses)
    model = jnp.hstack([tc1model, tc2model])
    #model = jnp.where(jnp.abs(model)<jnp.inf, model, -10000)
    numpyro.deterministic("model", model)
    numpyro.sample("obs", dist.Normal(loc=model, scale=error), obs=data)

#%%
kernel = numpyro.infer.NUTS(model, target_accept_prob=0.9)

#%%
nw, ns = 100*5, 100*5

#%%
mcmc = numpyro.infer.MCMC(kernel, num_warmup=nw, num_samples=ns)

#%%
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, data, error, extra_fields=('potential_energy',))

#%%
mcmc.print_summary()

#%%
samples = mcmc.get_samples()
meanmodel = jnp.mean(samples['model'], axis=0)

#%%
m1, m2 = meanmodel[:len(tc1)], meanmodel[len(tc1):]

#%%
plt.plot(tc1[1:], np.diff(tc1)-np.mean(np.diff(tc1)), '.')
plt.plot(tc2[1:], np.diff(tc2)-np.mean(np.diff(tc2)), '.')
plt.plot(m1[1:], np.diff(m1)-np.mean(np.diff(m1)), '-')
plt.plot(m2[1:], np.diff(m2)-np.mean(np.diff(m2)), '-')

#%%
import corner
import pandas as pd

#%%
keys = ["mass", "period", "ecc", "omega", "tic"]

#%%
def planet(samples, j):
    pdic = {}
    for k in keys:
        if k=='mass':
            pdic[k] = samples[k][:,j+1] / 3e-6
        else:
            pdic[k] = samples[k][:,j]
    return pdic
"""
