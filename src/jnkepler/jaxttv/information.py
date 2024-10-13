
__all__ = ["information", "scale_information", "observed_information", "hessian", "information_numpyrox"]

import jax.numpy as jnp
from jax import jacfwd, jacrev


def model_pdic(jttv, pdic, ms=1., lnmass=False):
    """transit times from parameter dict
    
        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters
            ms: stellar mass (solar unit)

        Returns:
            1D flattened model transit times
    
    """
    ecosw, esinw, period, tic, lnode, cosi \
        = pdic['ecosw'], pdic['esinw'], pdic['period'], pdic['tic'], pdic['lnode'], pdic['cosi'] 
    mass = jnp.exp(pdic['lnmass']) if lnmass else pdic['mass']
    elements = jnp.stack([period, ecosw, esinw, cosi, lnode, tic]).T 
    masses = jnp.hstack([ms, mass])
    return jttv.get_transit_times_obs(elements, masses)[0]


def information(jttv, pdic, keys):
    """compute Fisher information matrix for iid gaussian likelihood

        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters; keys must contain {ecosw, esinw, period, tic, lnode, cosi} and {mass or lnmass}
            keys: parameter keys for computing fisher matrix

        Returns:
            information matrix computed as grad.T Sigma_inv grad

    """
    assert {'ecosw', 'esinw', 'period', 'tic', 'lnode', 'cosi'}.issubset(pdic.keys()), "pdic keys must contain all of ecosw, esinw, period, tic, lnode, cosi."
    assert 'mass' in pdic.keys() or 'lnmass' in pdic.keys(), "pdic keys must contain either mass or lnmass."
    flag_lnmass = True if "lnmass" in keys else False
    assert set(keys).issubset({'ecosw', 'esinw', 'period', 'tic', 'lnode', 'cosi', 'lnmass' if flag_lnmass else 'mass'}), "pdic keys must a subsect of {ecosw, esinw, period, tic, lnode, cosi}+{mass or lnmass}"
    jacobian_pytree = jacrev(model_pdic, argnums=1)(jttv, pdic, lnmass=flag_lnmass)
    jacobian = jnp.hstack([jacobian_pytree[key] for key in keys])
    sigma_inv = jnp.diag(1. / jttv.errorobs_flatten**2)
    information_matrix = jacobian.T@sigma_inv@jacobian
    return information_matrix


def get_2d_matrix(p, param_order):
    """extract 2D matrix from pytree

        Args:
            p: pytree (as output from numpyro_ext.information)
            param_order: list of parameter keys

        Returns:
            2D array

    """
    coord_dict = {}
    N_par = 0
    size = len(p[param_order[0]][param_order[0]][0])

    for par in param_order:
        coord_dict[par] = jnp.arange(N_par, N_par + size)
        N_par += size

    arr_2D = jnp.zeros((N_par, N_par))
    for k1 in param_order:
        for k2 in param_order:
            arr_2D = arr_2D.at[jnp.ix_(coord_dict[k1], coord_dict[k2])].set(p[k1][k2])

    return arr_2D


def information_numpyrox(numpyro_model, pdic, **kwargs):
    """Fisher information from numpyro model using numpryo-ext

        Args:
            numpyro_model: numpyro model 
            pdic: dict containing parameters
            kwargs: additional arguments for numpyro model

        Returns:
            information matrix evaulated at pdic, list of site names

    """
    from numpyro_ext import information
    info_inv = information(numpyro_model, invert=True)(pdic, **kwargs)
    pnames = list(info_inv.keys())
    matrix = get_2d_matrix(info_inv, param_order=pnames)
    return matrix, pnames


def scale_information(matrix, param_bounds, keys):
    """get information matrix for scaled parameters
    
        Args:
            matrix: information matrix
            param_bounds: dict containing bounds for parameters, 0: lower, 1: upper
            keys: parameter keys (normally ['ecosw', 'esinw', 'mass', 'period', 'tic'])

        Returns:
            information matrix for scaled parameters scaled by 1/(upper - lower)

    
    """
    scaled_matrix = jnp.zeros_like(matrix)
    npl = len(param_bounds["mass"][0])
    for i,key1 in enumerate(keys):
        for j,key2 in enumerate(keys):
            scale_factor = jnp.einsum("i,j->ij", param_bounds[key1][1] - param_bounds[key1][0], param_bounds[key2][1] - param_bounds[key2][0])
            new_elements = matrix[npl*i:npl*(i+1),npl*j:npl*(j+1)] * scale_factor
            scaled_matrix = scaled_matrix.at[npl*i:npl*(i+1),npl*j:npl*(j+1)].set(new_elements)
    return scaled_matrix
    

def negative_log_likelihood(jttv, pdic, lnmass=False):
    """negative log likelihood (iid gaussian)
    
        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters

        Returns:
            negative log likelihood
    
    """
    transit_times = model_pdic(jttv, pdic, lnmass=lnmass)
    return 0.5 * jnp.sum( ((jttv.tcobs_flatten - transit_times) / jttv.errorobs_flatten)**2 )


def observed_information(jttv, pdic, keys):
    """compute observed Fisher information matrix (a.k.a. Hessian) for iid gaussian likelihood
    returns the same as hessian for keys=['ecosw', 'esinw', 'mass', 'period', 'tic'].

        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters

        Returns:
            observed information matrix computed as grad.T Sigma_inv grad

    """
    assert {'ecosw', 'esinw', 'period', 'tic', 'lnode', 'cosi'}.issubset(pdic.keys()), "pdic keys must contain all of ecosw, esinw, period, tic, lnode, cosi."
    assert 'mass' in pdic.keys() or 'lnmass' in pdic.keys(), "pdic keys must contain either mass or lnmass."
    flag_lnmass = True if "lnmass" in keys else False
    assert set(keys).issubset({'ecosw', 'esinw', 'period', 'tic', 'lnode', 'cosi', 'lnmass' if flag_lnmass else 'mass'}), "pdic keys must a subsect of {ecosw, esinw, period, tic, lnode, cosi}+{mass or lnmass}"

    # jacfwd fails for newton-raphson method
    from copy import deepcopy
    if jttv.transit_time_method != "interpolation":
        jttv_copy = deepcopy(jttv)
        jttv_copy.transit_time_method = "interpolation"
    else:
        jttv_copy = jttv

    hessian_pytree = jacfwd(jacrev(negative_log_likelihood, argnums=1), argnums=1)(jttv_copy, pdic, lnmass=flag_lnmass)
    
    return get_2d_matrix(hessian_pytree, keys)


def hessian(self, pdic):
    """compute hessian for iid gaussian likelihood; CURRENTLY WORKS ONLY FOR ['ecosw', 'esinw', 'mass', 'period', 'tic']
    for these keys, this function returns the same matirx as observed_hessian but is faster
    
        Args:
            pdic: parameter dictionary

        Returns:
            hessian (second derivative of the negative log likelihood)

    """
    from jnkepler.jaxttv.utils import initialize_jacobi_xv
    from jnkepler.jaxttv.findtransit import find_transit_times_kepler_all
    from jnkepler.jaxttv.symplectic import integrate_xv

    keys = ['ecosw', 'esinw', 'mass', 'period', 'tic']
    
    def negloglike(parr):
        # jacfwd fails for newton-raphson method, so use interpolate method
        npl = self.nplanet
        ecosw, esinw, mass, period, tic = parr[:npl], parr[npl:2*npl], parr[2*npl:3*npl], parr[3*npl:4*npl], parr[4*npl:5*npl]
        lnode, cosi = 0.*period, 0.*period
        elements = jnp.stack([period, ecosw, esinw, cosi, lnode, tic]).T 
        masses = jnp.hstack([1., mass])

        masses = jnp.hstack([1., mass])
        xjac0, vjac0 = initialize_jacobi_xv(elements, masses, self.t_start)
        times, xvjac = integrate_xv(xjac0, vjac0, masses, self.times, nitr=self.nitr_kepler)
        orbit_idx = self.pidx.astype(int) - 1
        tcobs1d = self.tcobs_flatten
        transit_times = find_transit_times_kepler_all(orbit_idx, tcobs1d, times, xvjac, masses, nitr=self.nitr_transit)
        return 0.5 * jnp.sum( ((self.tcobs_flatten - transit_times)/self.errorobs_flatten)**2 )

    parr = jnp.hstack([jnp.array(pdic[key]) for key in keys])
    hessian = jacfwd(jacrev(negloglike))(parr)
    
    return hessian