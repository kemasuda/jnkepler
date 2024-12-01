
__all__ = ["optim_svi", "fit_t_distribution"]

from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.initialization import init_to_value, init_to_sample
from scipy.stats import t as tdist
from scipy.stats import norm
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def optim_svi(numpyro_model, step_size, num_steps, p_initial=None, **kwargs):
    """optimization using Stochastic Variational Inference (SVI)

        Args:
            numpyro_model: numpyro model
            step_size: step size for optimization
            num_steps: # of steps for optimization
            p_initial: initial parameter set (dict); if None, use init_to_sample to initialize

        Returns:
            dict: dictionary containing optimized parameters

    """
    optimizer = numpyro.optim.Adam(step_size=step_size)

    if p_initial is None:
        guide = AutoLaplaceApproximation(
            numpyro_model, init_loc_fn=init_to_sample)
    else:
        guide = AutoLaplaceApproximation(
            numpyro_model, init_loc_fn=init_to_value(values=p_initial))

    # SVI object
    svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO(), **kwargs)

    # run the optimizer and get the posterior median
    svi_result = svi.run(random.PRNGKey(0), num_steps)
    params_svi = svi_result.params
    p_fit = guide.median(params_svi)

    return p_fit


def fit_t_distribution(y, plot=True, fit_mean=False):
    """fit Student's t distribution to a sample y

        Args:
            y: 1D array
            plot: if True, plot results
            fit_mean: if True, mean of the distribution is also fitted

        Returns: 
            dict: dictionary with the following keys:
                - lndf_loc: mean of log(dof)
                - lndf_scale: std of log(dof)
                - lnvar_loc: mean of log(variance)
                - lnvar_scale: std of log(variance)
                - mean_loc: mean of mean (if fitted)
                - mean_scale: std of mean (if fitted)

    """
    def model(y):
        logdf = numpyro.sample(
            "lndf", dist.Uniform(jnp.log(0.1), jnp.log(100)))
        logvar = numpyro.sample("lnvar", dist.Uniform(-2, 10))
        df = numpyro.deterministic("df", jnp.exp(logdf))
        v1 = numpyro.deterministic("v1", jnp.exp(logvar))
        if fit_mean:
            mean = numpyro.sample(
                "mean", dist.Uniform(-jnp.std(y), jnp.std(y)))
            numpyro.sample("obs", dist.StudentT(
                loc=mean, scale=jnp.sqrt(v1), df=df), obs=y)
        else:
            numpyro.sample("obs", dist.StudentT(
                scale=jnp.sqrt(v1), df=df), obs=y)

    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=500)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, y)
    mcmc.print_summary()

    samples = mcmc.get_samples()
    lndf, lnvar = np.mean(samples['lndf']), np.mean(samples['lnvar'])
    lndf_sd, lnvar_sd = np.std(samples['lndf']), np.std(samples['lnvar'])
    pout = {'lndf_loc': lndf, 'lndf_scale': lndf_sd,
            'lnvar_loc': lnvar, 'lnvar_scale': lnvar_sd}
    if fit_mean:
        mean, mean_sd = np.mean(samples['mean']), np.std(samples['mean'])
        pout['mean_loc'] = mean
        pout['mean_scale'] = mean_sd
    else:
        mean = 0.

    if plot:
        sd = np.std(y)
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        ax[1].set_yscale("log")
        ax[1].set_ylabel("PDF")
        ax[0].set_ylabel("CDF")
        ax[0].set_xlabel("residual / assigned error")
        ax[1].set_xlabel("residual / assigned error")
        ax[1].hist(y, histtype='step', lw=3, alpha=0.6,
                   density=True, color='gray')
        ymin, ymax = plt.gca().get_ylim()
        ax[1].set_ylim(ymin/5., ymax*1.5)
        x0 = np.linspace(-5, 5, 100)
        ax[1].plot(x0, norm(scale=sd).pdf(x0), lw=1, color='C0', ls='dashed',
                   label='normal, $\mathrm{SD}=%.2f$' % sd)
        ax[1].plot(x0, norm.pdf(x0), lw=1, color='C0', ls='dotted',
                   label='normal, $\mathrm{SD}=1$')
        ax[1].plot(x0, tdist(loc=mean, scale=np.exp(lnvar*0.5), df=np.exp(lndf)).pdf(x0),
                   label='Student\'s t\n(lndf=%.2f, lnvar=%.2f, mean=%.2f)' % (lndf, lnvar, mean))

        ysum = np.ones_like(y)
        hist, edge = np.histogram(y, bins=len(y))
        ax[0].plot(np.r_[x0[0], edge[0], edge[:-1], edge[-1], x0[-1]],
                   np.r_[0, 0, np.cumsum(hist)/len(y), 1, 1], lw=3, alpha=0.6, color='gray')
        ax[0].plot(x0, norm(loc=0, scale=sd).cdf(x0), lw=1, color='C0', ls='dashed',
                   label='normal, $\mathrm{SD}=%.2f$' % sd)
        ax[0].plot(x0, norm.cdf(x0), lw=1, color='C0', ls='dotted',
                   label='normal, $\mathrm{SD}=1$')
        ax[0].plot(x0, tdist(loc=mean, scale=np.exp(lnvar*0.5), df=np.exp(lndf)).cdf(x0),
                   label='Student\'s t\n(lndf=%.2f, lnvar=%.2f, mean=%.2f)' % (lndf, lnvar, mean))
        ax[0].legend(loc='upper left', fontsize=14)

    return pout
