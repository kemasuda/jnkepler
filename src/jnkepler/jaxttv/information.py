
__all__ = ["information", "observed_information", "hessian"]

import jax.numpy as jnp
from jax import jacfwd, jacrev, jit


def model_pdic(jttv, pdic):
    """transit times from parameter dict
    
        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters

        Returns:
            1D flattened model transit times
    
    """
    ecosw, esinw, mass, period, tic, lnode, cosi \
        = pdic['ecosw'], pdic['esinw'], pdic['mass'], pdic['period'], pdic['tic'], pdic['lnode'], pdic['cosi'] 
    elements = jnp.stack([period, ecosw, esinw, lnode, cosi, tic]).T 
    masses = jnp.hstack([1., mass])
    return jttv.get_ttvs(elements, masses)[0]


def negative_log_likelihood(jttv, pdic):
    """negative log likelihood (iid gaussian)
    
        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters

        Returns:
            negative log likelihood
    
    """
    transit_times = model_pdic(jttv, pdic)
    return 0.5 * jnp.sum( ((jttv.tcobs_flatten - transit_times) / jttv.errorobs_flatten)**2 )


def information(jttv, pdic, keys=['ecosw', 'esinw', 'mass', 'period', 'tic']):
    """compute Fisher information matrix for iid gaussian likelihood

        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters

        Returns:
            information matrix computed as grad.T Sigma_inv grad

    """
    jacobian_pytree = jacrev(model_pdic, argnums=1)(jttv, pdic)
    jacobian = jnp.hstack([jacobian_pytree[key] for key in keys])
    sigma_inv = jnp.diag(1. / jttv.errorobs_flatten**2)
    information_matrix = jacobian.T@sigma_inv@jacobian
    return information_matrix


def get_2d_matrix(p, param_order=['ecosw', 'esinw', 'mass', 'period', 'tic']):
    """extract 2D matrix from pytree
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
    

def observed_information(jttv, pdic, keys=['ecosw', 'esinw', 'mass', 'period', 'tic']):
    """compute observed Fisher information matrix (a.k.a. Hessian) for iid gaussian likelihood

        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters

        Returns:
            observed information matrix computed as grad.T Sigma_inv grad

    """
    # jacfwd fails for newton-raphson method
    from copy import deepcopy
    if jttv.transit_time_method != "interpolation":
        jttv_copy = deepcopy(jttv)
        jttv_copy.transit_time_method = "interpolation"
    else:
        jttv_copy = jttv

    hessian_pytree = jacfwd(jacrev(negative_log_likelihood, argnums=1), argnums=1)(jttv_copy, pdic)
    
    return get_2d_matrix(hessian_pytree)


def hessian(self, pdic, keys=['ecosw', 'esinw', 'mass', 'period', 'tic']):
    """compute hessian for iid gaussian likelihood; DOES NOT WORK FOR OTHER KEYS
    
        Args:
            pdic: parameter dictionary

        Returns:
            hessian (second derivative of the negative log likelihood)

    """
    from jnkepler.jaxttv.utils import initialize_jacobi_xv
    from jnkepler.jaxttv.findtransit import find_transit_times_kepler_all
    from jnkepler.jaxttv.symplectic import integrate_xv
    
    def negloglike(parr):
        # jacfwd fails for newton-raphson method, so use interpolate method
        npl = self.nplanet
        ecosw, esinw, mass, period, tic = parr[:npl], parr[npl:2*npl], parr[2*npl:3*npl], parr[3*npl:4*npl], parr[4*npl:5*npl]
        lnode, cosi = 0.*period, 0.*period
        elements = jnp.stack([period, ecosw, esinw, lnode, cosi, tic]).T 
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