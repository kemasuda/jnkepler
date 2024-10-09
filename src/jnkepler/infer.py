
__all__ = ["run_svi_optim", "ttv_default_parameter_bounds", "ttv_optim_curve_fit", "scale_pdic", "unscale_pdic"]

from jax import random, jacrev
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.initialization import init_to_value, init_to_sample
import numpy as np
import jax.numpy as jnp
from scipy.optimize import curve_fit
from copy import deepcopy
import time
from jnkepler.jaxttv.utils import params_to_elements


def run_svi_optim(numpyro_model, step_size, num_steps, p_initial=None):
    """SVI optimizer

        Args:
            numpyro_model: numpyro model
            step_size: step size for optimization
            num_steps: # of steps for optimization
            p_initial: initial parameter set (dict); if None, use init_to_sample to initialize

        Returns:
            p_fit: optimized parameter set

    """
    optimizer = numpyro.optim.Adam(step_size=step_size)
    
    if p_initial is None:
        guide = AutoLaplaceApproximation(numpyro_model, init_loc_fn=init_to_sample)
    else:
        guide = AutoLaplaceApproximation(numpyro_model, init_loc_fn=init_to_value(values=p_initial))

    # Create a Stochastic Variational Inference (SVI) object with NumPyro
    svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO())

    # Run the optimiser and get the median parameters
    svi_result = svi.run(random.PRNGKey(0), num_steps)
    params_svi = svi_result.params
    p_fit = guide.median(params_svi)

    return p_fit


def ttv_default_parameter_bounds(jttv, dtic=0.05, dp_frac=1e-2, emax=0.2, mmin=1e-7, mmax=1e-3):
    """get parameter bounds for TTV optimization

        Args:
            jttv: JaxTTV object

        Returns:
            dictionary of (lower bound array, upper bound array)

    """
    npl = jttv.nplanet
    t0_guess = np.array([tcobs_[0] for tcobs_ in jttv.tcobs])
    p_guess = np.array(jttv.p_init)

    ones = np.ones(npl)
    evector_shift = 1e-2 # avoid (0,0) for initial (ecosw, esinw)
    param_bounds = {
        "tic": [t0_guess - dtic, t0_guess + dtic],
        "period": [p_guess * (1 - dp_frac), p_guess * (1 + dp_frac)],
        "ecosw": [-emax * ones, emax * ones - evector_shift],
        "esinw": [-emax * ones, emax * ones - evector_shift],
        "lnmass": [np.log(mmin) * ones, np.log(mmax) * ones],
        "mass": [mmin * ones, mmax * ones],
    }

    return param_bounds


def scale_pdic(pdic, param_bounds):
    """scale parameters using bounds
    
        Args:
            pdic: dict of physical parameters
            param_bounds: dictionary of (lower bound array, upper bound array)

        Returns:
            dict of scaled parameters
    
    """
    pdic_scaled = {}
    for key in param_bounds.keys():
        pdic_scaled[key+"_scaled"] = (pdic[key] - param_bounds[key][0]) / (param_bounds[key][1] - param_bounds[key][0])
    return pdic_scaled


def unscale_pdic(pdic_scaled, param_bounds):
    """unscale parameters using bounds
    
        Args:
            pdic: dict of scaled parameters
            param_bounds: dictionary of (lower bound array, upper bound array)

        Returns:
            dict of physical parameters in original scales
    
    """
    pdic = {}
    for key in param_bounds.keys():
        pdic[key] = param_bounds[key][0] + (param_bounds[key][1] - param_bounds[key][0]) * pdic_scaled[key+"_scaled"]
    return pdic


def ttv_optim_curve_fit(jttv, param_bounds_, p_init=None, jac=False):
    """simple TTV fit using scipy.curve_fit with bounds

        Args:
            jttv: JaxTTV object
            param_bounds_: bounds for parameters, 0: lower, 1: upper
            p_init: initial parameter values (if None, center of lower/upper bounds)
            jac: if True jacrev(model) is used

        Returns:
            JaxTTV parameter array 

    """
    npl = jttv.nplanet 
    param_bounds = deepcopy(param_bounds_)

    if "cosi" not in param_bounds.keys() or "lnode" not in param_bounds.keys():
        print ("# bounds for cosi/lnode not provided: assuming coplanar orbits...")
        ones = np.ones_like(param_bounds["period"][0])
        param_bounds["cosi"] = [-1e-6 * ones, 1e-6 * ones]
        param_bounds["lnode"] = [-1e-6 * ones, 1e-6 * ones]
    
    keys = ['period', 'ecosw', 'esinw', 'cosi', 'lnode', 'tic']
    params_lower = np.hstack([np.array([[param_bounds[key][0][j] for key in keys] for j in range(npl)]).ravel(), param_bounds["lnmass"][0]])
    params_upper = np.hstack([np.array([[param_bounds[key][1][j] for key in keys] for j in range(npl)]).ravel(), param_bounds["lnmass"][1]])
    bounds = (params_lower, params_upper)

    if p_init is None:
        p_init = 0.5 * (params_lower + params_upper)

    model = lambda p: jttv.get_ttvs(*params_to_elements(p, npl))[0]
    func = lambda x, *params: model(jnp.array(params))
    objective = lambda p: jnp.sum( (model(p) - jttv.tcobs_flatten)**2 / jttv.errorobs_flatten**2 )
    if jac:
        jacmodel = jacrev(model)
        jacfunc = lambda x, *params: jacmodel(jnp.array(params))
    else:
        jacfunc = None

    print ("# running least squares optimization...")
    start_time = time.time()
    popt, pcov = curve_fit(func, None, jttv.tcobs_flatten, p0=p_init, sigma=jttv.errorobs_flatten, bounds=bounds, jac=jacfunc)
    print ("# objective function: %.2f --> %.2f (%d data)"%(objective(p_init), objective(popt), len(jttv.tcobs_flatten)))
    print ("# elapsed time: %.1f sec" % (time.time()-start_time))

    return popt