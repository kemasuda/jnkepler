""" routines to compute transit light curves
"""
__all__ = ["get_xvast_map", "compute_nbody_flux",
           "compute_nbody_flux_nooverlap"]

import jax.numpy as jnp
from jax import vmap
rsun_au = 0.00465047


def get_xvast_map(xcm, vcm, pidxarr):
    """astrocentric positions and velocities at transit centers

        Args:
            xcm: center-of-mass positions (Nbody, xyz)
            vcm: center-of-mass velocities (Nbody, xyz)
            pidxarr: array of planet numbers starting from 1 (1D array)

        Returns:
            array: astrocentric positions and velocities at transit centers in the sky plane (Ntransit, xy)

    """
    def xvast_orbit(xcm, vcm, j):
        return xcm[j, :2] - xcm[0, :2], vcm[j, :2] - vcm[0, :2]
    xvast_map = vmap(xvast_orbit, (0, 0, 0), (0))
    return xvast_map(xcm, vcm, pidxarr)


def compute_relative_flux_loss(barr, rarr, u1, u2):
    """compute relative flux loss using exoplanet_core

        Args:
            barr: array of planet-star distances in the sky plane normalized by the stellar radius
            rarr: array of planet-star radius ratios
            u1, u2: quadratic limb-darkening coefficients

        Returns:
            array: relative flux loss

    """
    try:
        from jaxoplanet.core.limb_dark import light_curve
    except ImportError:
        raise ImportError(
            "The 'jaxoplanet' package is required for using the NbodyTransit module. "
            "Please install it using 'pip install jaxoplanet'. "
            "For more details, visit https://jax.exoplanet.codes/en/latest/"
        )
    return light_curve([u1, u2], barr, rarr)


def compute_nbody_flux(rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, times, times_transit_idx, times_planet_idx):
    """compute light curve given N-body model

        Note:
            This function can handle simultaneous transits but is slightly slower than `compute_nbody_flux_nooverlap.`
            Overlap between the planets during a transit is not yet considered.

        Args:
            rstar: stellar radius (solar unit)
            prad: planet-to-star radius ratios (Nplanet,)
            u1, u2: quadratic limb-darkening coefficients
            tc: transit times (Ntransit,)
            xsky_tc: astrocentric posistions in the sky plane at transit centers (Ntransit,)
            xsky_tc: astrocentric velocities in the sky plane at transit centers (Ntransit,)
            times: times at which fluxes are evaluated (Ntime,)
            times_transit_idx: indices of nearest transit centers (Nplanet, Ntime)
            times_planet_idx: indices of planets (Nplanet, Ntime)

        Returns:
            array: relative flux loss

    """
    xsky_au = xsky_tc[times_transit_idx] + vsky_tc[times_transit_idx] * \
        (times - tc[times_transit_idx])[:, :, None]
    barr_au = jnp.sqrt(jnp.sum(xsky_au**2, axis=2))
    barr = barr_au / (rsun_au * rstar)
    rarr = prad[times_planet_idx]
    return jnp.sum(compute_relative_flux_loss(barr, rarr, u1, u2), axis=0)


def compute_nbody_flux_nooverlap(rstar, prad, u1, u2, tc, xsky_tc, vsky_tc, times, times_transit_idx, times_planet_idx):
    """ compute light curve given N-body model

        Note:
            This function assumes that the data do not include simultaneous transits of two planets.

        Args:
            rstar: stellar radius (solar unit)
            prad: planet-to-star radius ratios (Nplanet,)
            u1, u2: quadratic limb-darkening coefficients
            tc: transit times (Ntransit,)
            xsky_tc: astrocentric posistions in the sky plane at transit centers (Ntransit,)
            xsky_tc: astrocentric velocities in the sky plane at transit centers (Ntransit,)
            times: times at which fluxes are evaluated (Ntime,)
            times_transit_idx: indices of nearest transit centers (Ntime,)
            times_planet_idx: indices of planets (Ntime,)

        Returns:
            array: relative flux loss

    """
    xsky_au = xsky_tc[times_transit_idx] + vsky_tc[times_transit_idx] * \
        (times - tc[times_transit_idx])[:, None]
    barr_au = jnp.sqrt(jnp.sum(xsky_au**2, axis=1))
    barr = barr_au / (rsun_au * rstar)
    rarr = prad[times_planet_idx]
    return compute_relative_flux_loss(barr, rarr, u1, u2)
