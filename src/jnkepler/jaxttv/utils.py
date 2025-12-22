"""
Internal utilities for parameter conversion, initialization, and diagnostics.
"""
__all__ = [
    "initialize_jacobi_xv", "get_energy_diff", "get_energy_diff_jac",
    "params_to_elements", "elements_to_pdic", "convert_elements", "findidx_map", "params_to_dict", "dict_to_params", "em_to_dict"
]

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, config
from .symplectic import kepler_step_map
from .conversion import m_to_u, tic_to_m, tic_to_u, elements_to_xv, xv_to_elements, G, xvjac_to_xvcm
config.update('jax_enable_x64', True)


M_earth = 3.0034893e-6


def params_to_dict(params, npl, keys):
    """convert 1D parameter array into parameter dict

        Args:
            parameter array: [arr for key1, arr for key2, ...] where len(arr) is the number of planets
            npl: number of planets
            keys: parameter keys [key1, key2, ...]

        Returns:
            parameter dict

    """
    pdic = {}

    for i, key in enumerate(keys):
        pdic[key] = params[i*npl:(i+1)*npl]

    return pdic


def dict_to_params(pdic, npl, keys):
    """
    Inverse of `params_to_dict` when each value has length `npl`.

    Args:
        pdic (Mapping[str, array_like]):
            Parameter dict, e.g. {'a': arr(len=npl), 'b': arr(len=npl), ...}.
        keys (Sequence[str]):
            Order of parameters; concatenation follows this order.

    Returns:
        ndarray or jax.numpy.ndarray:
            1D parameter array of length len(keys) * npl.

    Raises:
        KeyError: if a key in `keys` is missing from `pdic`.
        ValueError: if value lengths are inconsistent across keys.
    """
    arrs = []
    for k in keys:
        if k not in pdic:
            raise KeyError(f"Key {k!r} not in pdic.")
        a = np.asarray(pdic[k]).ravel()
        if a.size != npl:
            raise ValueError(f"Length mismatch at {k!r}: {a.size} != {npl}")
        arrs.append(a)
    params = np.concatenate(arrs, axis=0)
    return params


def em_to_dict(elements, masses):
    """convert arrays of elements and masses in v0.1.0 to parameter dict

        Note:
            This function is mainly for running tests; no longer needed for v>=0.2.

        Args:
            elements: elements (JaxTTV format)
            masses: masses of star + planets (solar units)

        Returns:
            parameter dict

    """
    pdic = {}
    for k, key in enumerate(["period", "ecosw", "esinw", "cosi", "lnode", "tic"]):
        pdic[key] = elements[:, k]
    pdic['pmass'] = masses[1:]
    return pdic


def initialize_jacobi_xv(par_dict, t_epoch):
    """compute initial position/velocity from parameter dict

        Note:
            Here the elements are interpreted as Jacobi elements using the total interior mass (see Section 2.2 of Rein & Tamayo 2015).

        Args:
            par_dict: parameter dictionary that needs to contain
                - either (ecosw, esinw) or (e, omega)
                - cosi (set to be 0 if not specified)
                - lnode (set to be 0 if not specified)
                - either (time of inferior conjunction) or (mean anomaly)
                - stellar mass (set to be 1 if not specified), solar unit
                - either (planetary mass) or (ln planetary mass), solar unit, former is used when both are provided

            t_epoch: epoch at which elements are defined

        Returns:
            tuple:
                - Jacobi positions at t_epoch (Norbit, xyz)
                - Jacobi velocities at t_epoch (Norbit, xyz)
                - masses: 1D array of stellar and planetary masses (Nbody,)

    """
    keys = par_dict.keys()

    period = par_dict["period"]

    if "ecosw" in keys and "esinw" in keys:
        ecosw, esinw = par_dict['ecosw'], par_dict['esinw']
        ecc = jnp.sqrt(ecosw**2 + esinw**2)
        omega = jnp.arctan2(esinw, ecosw)
    elif "ecc" in keys and "omega" in keys:
        ecc, omega = par_dict["ecc"], par_dict["omega"]
    else:
        raise ValueError(
            "Either (ecosw, esinw) or (ecc, omega) needs to be provided.")

    if "cosi" in keys:
        inc = jnp.arccos(par_dict["cosi"])
    else:
        inc = jnp.arccos(period * 0.)

    if "lnode" in keys:
        lnode = par_dict["lnode"]
    else:
        lnode = period * 0.

    if "tic" in keys:
        ma = tic_to_m(par_dict["tic"], period, ecc, omega, t_epoch)
    elif "ma" in keys:
        ma = par_dict["ma"]
    else:
        raise ValueError(
            "Either tic (time of inf. conjunction) or ma (mean anom.) needs to be provided.")
    u = m_to_u(ma, ecc)  # eccentric anomaly

    if "smass" in keys:
        smass = par_dict["smass"]
    else:
        smass = 1.  # in this case pmass should be considered as planet-to-star mass ratio

    if "pmass" in keys:
        masses = jnp.hstack([smass, par_dict['pmass']])
    elif "lnpmass" in keys:
        masses = jnp.hstack([smass, jnp.exp(par_dict['lnpmass'])])
    else:
        raise ValueError(
            "Either pmass (solar unit) or lnpmass needs to be provided.")

    xjac, vjac = [], []
    for j in range(len(period)):
        xj, vj = elements_to_xv(
            period[j], ecc[j], inc[j], omega[j], lnode[j], u[j], jnp.sum(masses[:j+2]))
        xjac.append(xj)
        vjac.append(vj)

    return jnp.array(xjac), jnp.array(vjac), masses


@jit
def get_energy(x, v, masses):
    """compute total energy of the system in CM frame

        Args:
            x: CM positions (Nbody, xyz)
            v: CM velocities (Nbody, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            total energy in units of Msun*(AU/day)^2

    """
    K = jnp.sum(0.5 * masses * jnp.sum(v*v, axis=1))
    X = x[:, None] - x[None, :]
    M = masses[:, None] * masses[None, :]
    U = -G * jnp.sum(M * jnp.tril(1./jnp.sqrt(jnp.sum(X*X, axis=2)), k=-1))
    return K + U


# map along the 1st axes of x and v (Nstep)
get_energy_map = jit(vmap(get_energy, (0, 0, None), 0))


@jit
def get_energy_diff(xva, masses):
    """compute fractional energy change given integration result

        Args:
            xva: posisions, velocities, accelerations in CoM frame (Nstep, x or v or a, Norbit, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            fractional change in total energy

    """
    _xva = jnp.array([xva[0, :, :, :], xva[-1, :, :, :]])
    etot = get_energy_map(_xva[:, 0, :, :], _xva[:, 1, :, :], masses)
    return etot[1]/etot[0] - 1.


@jit
def get_energy_diff_jac(xvjac, masses, dt):
    """compute fractional energy change given integration result

        Args:
            xvjac: Jacobi posisions and velocities (Nstep, x or v, Norbit, xyz)
            masses: masses of the bodies (Nbody,)

        Returns:
            fractional change in total energy

    """
    xvjac_ends = jnp.array([xvjac[0], xvjac[-1]])
    xjac_ends_correct, vjac_ends_correct = kepler_step_map(
        xvjac_ends[:, 0, :, :], xvjac_ends[:, 1, :, :], masses, dt)
    xcm, vcm = xvjac_to_xvcm(xjac_ends_correct, vjac_ends_correct, masses)
    etot = get_energy_map(xcm, vcm, masses)
    return etot[1]/etot[0] - 1.


def elements_to_pdic(elements, masses, outkeys=None, force_coplanar=True):
    """convert JaxTTV elements/masses into dictionary

        Note:
            This function is for v<0.2.

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
    """convert JaxTTV parameter array into element and mass arrays

        Args:
            params: JaxTTV parameter array
            npl: number of orbits (planets)

        Returns:
            tuple:
                - Jacobi orbital elements (period, ecosw, esinw, cosi, \Omega, T_inf_conjunction)
                - ln(masses) of the bodies (Nbody,)

    """
    elements = jnp.array(params[:-npl].reshape(npl, -1))
    masses = jnp.exp(jnp.hstack([0, params[-npl:]]))
    return elements, masses


def convert_elements(par_dict, t_epoch, WHsplit=False):
    """convert JaxTTV elements into more normal sets of parameters

        Args:
            par_dict: parameter dict
            t_epoch: epoch at which elements are defined
            WHsplit: elements are converted to coordinates assuming Wisdom-Holman splitting. This should be True when the output is used for TTVFast.

        Returns:
            tuple:
                 - array: (semi-major axis, period, eccentricity, inclination, argument of periastron, longitude of ascending node, mean anomaly) x (orbits), angles are in radians
                 - mass array

    """
    xjac, vjac, masses = initialize_jacobi_xv(par_dict, t_epoch)

    if WHsplit:
        # for H_Kepler defined in WH splitting (i.e. TTVFast)
        ki = G * masses[0] * jnp.cumsum(masses)[1:] / \
            jnp.hstack([masses[0], jnp.cumsum(masses)[1:][:-1]])
    else:
        # total interior mass
        ki = G * jnp.cumsum(masses)[1:]

    return xv_to_elements(xjac, vjac, ki), masses


def findidx_map(arr1, arr2):
    """pick up elements of arr1 nearest to each element in arr2

        Args:
            arr1: array from which elements are picked up
            arr2: array of the values for which nearest matches are searched

        Returns:
            indices of arr1 nearest to each element in arr2

    """
    def func(arr1, val): return jnp.argmin(jnp.abs(arr1 - val))
    func_map = jit(vmap(func, (None, 0), 0))
    return func_map(arr1, arr2)
