#%%
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from .markley import get_E
from jax.config import config
config.update('jax_enable_x64', True)

#%%
BIG_G = 2.959122082855911e-4


def initialize_from_elements(elements, masses, t_epoch):
    """ compute initial position/velocity from JaxTTV elements
        here the elements are interpreted as Jacobi using the total interior mass (see Rein & Tamayo 2015)

        Args:
            elements: Jacobi orbital elements (period, ecosw, esinw, cosi, \Omega, T_inf_conjunction)
            masses: masses of the bodies (Nbody,)
            t_epoch: epoch at which elements are defined

        Returns:
            Jacobi positions and velocities at t_epoch (Norbit, xyz)

    """
    xjac, vjac = [], []
    for j in range(len(elements)):
        #porb, ecc, inc, omega, lnode, tic = elements[j]
        porb, ecosw, esinw, cosi, lnode, tic = elements[j]
        ecc = jnp.sqrt(ecosw**2 + esinw**2)
        omega = jnp.arctan2(esinw, ecosw)
        inc = jnp.arccos(cosi)

        u = tic_to_u(tic, porb, ecc, omega, t_epoch)
        xj, vj = elements_to_xvrel(porb, ecc, inc, omega, lnode, u, BIG_G*jnp.sum(masses[:j+2]))
        xjac.append(xj)
        vjac.append(vj)

    return jnp.array(xjac), jnp.array(vjac)

@jit
def get_energy(x, v, masses):
    """ compute total energy of the system in CM frame

        Args:
            x: CM positions (Nbody, xyz)
            v: CM velocities (Nbody, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            total energy in units of Msun*(AU/day)^2

    """
    K = jnp.sum(0.5 * masses * jnp.sum(v*v, axis=1))
    X = x[:,None] - x[None,:]
    M = masses[:,None] * masses[None,:]
    U = -BIG_G * jnp.sum(M * jnp.tril(1./jnp.sqrt(jnp.sum(X*X, axis=2)), k=-1))
    return K + U

get_energy_vmap = jit(vmap(get_energy, (0,0,None), 0))

@jit
def get_ediff(xva, masses):
    """ compute fractional energy change given integration result

        Args:
            xva: posisions, velocities, accelerations in CoM frame (Nstep, x or v or a, Norbit, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            fractional change in total energy

    """
    _xva = jnp.array([xva[0,:,:,:], xva[-1,:,:,:]])
    etot = get_energy_vmap(_xva[:,0,:,:], _xva[:,1,:,:], masses)
    return etot[1]/etot[0] - 1.

@jit
def get_acm(x, masses):
    """ compute acceleration given position, velocity, mass

        Args:
            x: positions in CoM frame (Norbit, xyz)
            masses: masses of the bodies (Nbody)

        Returns:
            a: accelerations (Norbit, xyz)

    """
    xjk = jnp.transpose(x[:,None] - x[None, :], axes=[0,2,1])
    x2jk = jnp.sum(xjk * xjk, axis=1)[:,None,:]
    x2jk = jnp.where(x2jk!=0., x2jk, jnp.inf)
    x2jkinv = 1. / x2jk

    x2jkinv1p5 = x2jkinv * jnp.sqrt(x2jkinv)
    Xjk = - xjk * x2jkinv1p5

    a = BIG_G * jnp.dot(Xjk, masses)

    return a

# map along the 1st axis of x
geta_map = vmap(get_acm, (0,None), 0)


def get_elements(params, nplanet, t_epoch, WHsplit=False):
    """ convert JaxTTV parameter array into more normal sets of parameters

        Args:
            params: JAX TTV parameter array
            nplanet: number of orbits (planets)
            t_epoch: epoch at which elements are defined
            WHsplit: elements are converted to coordinates assuming Wisdom-Holman splitting. This should be True when the output is used for TTVFast.

        Returns:
            (semi-major axis, period, eccentricity, inclination, argument of periastron, longitude of ascending node, mean anomaly) x (orbits)

            angles are in radians

    """
    elements, masses = params_to_elements(params, nplanet)
    xjac, vjac = initialize_from_elements(elements, masses, t_start)

    if WHsplit:
        # for H_Kepler defined in WH splitting (i.e. TTVFast)
        ki = BIG_G * masses[0] * jnp.cumsum(masses)[1:] / jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])
    else:
        # total interior mass
        ki = BIG_G * jnp.cumsum(masses)[1:]

    return xv_to_elements(xjac, vjac, ki)


def findidx_map(arr1, arr2):
    """ pick up elements of arr1 nearest to each element in arr2

        Args:
            arr1: array from which elements are picked up
            arr2: array of the values for which nearest matches are searched 

        Returns:
            indices of arr1 nearest to each element in arr2

    """
    func = lambda arr1, val: jnp.argmin(jnp.abs(arr1 - val))
    func_map = jit(vmap(func, (None,0), 0))
    return func_map(arr1, arr2)
