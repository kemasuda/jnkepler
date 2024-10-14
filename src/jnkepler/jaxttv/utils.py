__all__ = [
    "initialize_jacobi_xv", "get_energy_diff", "get_energy_diff_jac",
    "params_to_elements", "elements_to_pdic", "convert_elements", "findidx_map"
]

import jax.numpy as jnp
from jax import jit, vmap, config
from jax.lax import scan
from .markley import get_E
from .conversion import tic_to_u, elements_to_xv, xv_to_elements, BIG_G, xvjac_to_xvcm
#from jax.config import config
config.update('jax_enable_x64', True)

#%%
M_earth = 3.0034893e-6


def initialize_jacobi_xv(elements, masses, t_epoch):
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
        porb, ecosw, esinw, cosi, lnode, tic = elements[j]
        ecc = jnp.sqrt(ecosw**2 + esinw**2)
        omega = jnp.arctan2(esinw, ecosw)
        inc = jnp.arccos(cosi)

        u = tic_to_u(tic, porb, ecc, omega, t_epoch)
        xj, vj = elements_to_xv(porb, ecc, inc, omega, lnode, u, jnp.sum(masses[:j+2]))
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

# map along the 1st axes of x and v (Nstep)
get_energy_map = jit(vmap(get_energy, (0,0,None), 0))

@jit
def get_energy_diff(xva, masses):
    """ compute fractional energy change given integration result

        Args:
            xva: posisions, velocities, accelerations in CoM frame (Nstep, x or v or a, Norbit, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            fractional change in total energy

    """
    _xva = jnp.array([xva[0,:,:,:], xva[-1,:,:,:]])
    etot = get_energy_map(_xva[:,0,:,:], _xva[:,1,:,:], masses)
    return etot[1]/etot[0] - 1.


'''
@jit
def get_energy_diff_jac(xvjac, masses):
    """ compute fractional energy change given integration result

        Args:
            xvjac: Jacobi posisions and velocities (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            fractional change in total energy

    """
    _xvjac = jnp.array([xvjac[0], xvjac[-1]])
    _xcm, _vcm = xvjac_to_xvcm(_xvjac, masses)
    etot = get_energy_map(_xcm, _vcm, masses)
    return etot[1]/etot[0] - 1.
'''


from .symplectic import kepler_step_map
@jit
def get_energy_diff_jac(xvjac, masses, dt):
    """ compute fractional energy change given integration result

        Args:
            xvjac: Jacobi posisions and velocities (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            fractional change in total energy

    """
    xvjac_ends = jnp.array([xvjac[0], xvjac[-1]])
    xjac_ends_correct, vjac_ends_correct = kepler_step_map(xvjac_ends[:,0,:,:], xvjac_ends[:,1,:,:], masses, dt)
    xcm, vcm = xvjac_to_xvcm(xjac_ends_correct, vjac_ends_correct, masses)
    etot = get_energy_map(xcm, vcm, masses)
    return etot[1]/etot[0] - 1.


def elements_to_pdic(elements, masses, outkeys=None, force_coplanar=True):
    """ convert JaxTTV elements/masses into dictionary

        Args:
            elements: Jacobi orbital elements (period, ecosw, esinw, cosi, \Omega, T_inf_conjunction)
            masses: masses of the bodies (Nbody,)
            outkeys: if specified only include these keys in the output
            force_coplanar: if True, set incl=pi/2 and lnode=0

        Returns:
            dicionary of the parameters

    """
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
    """ convert JaxTTV parameter array into element and mass arrays

        Args:
            params: JaxTTV parameter array
            npl: number of orbits (planets)

        Returns:
            elements: Jacobi orbital elements (period, ecosw, esinw, cosi, \Omega, T_inf_conjunction)
            masses: masses of the bodies (Nbody,)

    """
    elements = jnp.array(params[:-npl].reshape(npl, -1))
    masses = jnp.exp(jnp.hstack([0, params[-npl:]]))
    return elements, masses


def convert_elements(elements, masses, t_epoch, WHsplit=False):
    """ convert JaxTTV elements into more normal sets of parameters

        Args:
            params: JAX TTV parameter array
            nplanet: number of orbits (planets)
            t_epoch: epoch at which elements are defined
            WHsplit: elements are converted to coordinates assuming Wisdom-Holman splitting. This should be True when the output is used for TTVFast.

        Returns:
            (semi-major axis, period, eccentricity, inclination, argument of periastron, longitude of ascending node, mean anomaly) x (orbits)

            angles are in radians

    """
    xjac, vjac = initialize_jacobi_xv(elements, masses, t_epoch)

    if WHsplit:
        # for H_Kepler defined in WH splitting (i.e. TTVFast)
        ki = BIG_G * masses[0] * jnp.cumsum(masses)[1:] / jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])
    else:
        # total interior mass
        ki = BIG_G * jnp.cumsum(masses)[1:]

    return xv_to_elements(xjac, vjac, ki), masses


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
