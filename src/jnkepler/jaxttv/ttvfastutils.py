"""
Utilities for interfacing with TTVFast models.

This module provides helper functions for constructing TTVFast-compatible parameter sets and model objects. 
These functions are intended solely for comparison, validation, and conversion purposes, and
are not used in the core JAX-based modeling or inference pipelines.
"""
__all__ = ["params_for_ttvfast", "get_ttvfast_model",
           "get_ttvfast_model_rv", "get_ttvfast_model_all"]

import numpy as np
import pandas as pd
from jax import jit, vmap
from .utils import convert_elements
from ..nbodytransit.nbodytransit import b_to_cosi


def params_for_ttvfast(samples, t_epoch, num_planets, WHsplit=True, angles_in_degrees=True,
                       names=["period", "eccentricity", "inclination", "argument", "longnode", "mean_anomaly"]):
    """convert JaxTTV samples into TTVFast (or other) format

        Args:
            samples: mcmc.get_samples()
            t_epoch: time at which osculating elements are defined
            num_planets: number of planets
            WHsplit: True for TTVFast; False for e.g. REBOUND (cf. Section 2.2 of Rein & Tamayo 2015, MNRAS 452, 376)
            angles_in_degrees: If True, angles are returned in degrees

        Returns:
            dataframe containing parameters

    """
    def func(pdic):
        if 'b' in pdic.keys() and 'cosi' not in pdic.keys():
            cosi = b_to_cosi(pdic['b'], pdic['period'], pdic['ecosw'],
                             pdic['esinw'], pdic['srad'], pdic['smass'])
            pdic['cosi'] = cosi
        return convert_elements(pdic, t_epoch, WHsplit=WHsplit)
    convert_elements_map = jit(vmap(func, (0,), 0))
    elements, masses = convert_elements_map(samples)

    pdic = {}
    for j in range(num_planets):
        pdic['planet_mass%d' % j] = masses[:, j+1]
        for i, n in enumerate(names):
            pdic[n+"%d" % j] = elements[:, i+1, j]
    pdic['star_mass'] = masses[:, 0]
    pdic['num_planets'] = num_planets
    df = pd.DataFrame(data=pdic)

    if angles_in_degrees:
        for key in df.keys():
            if "inclination" in key or "argument" in key or "node" in key or "anomaly" in key:
                df[key] = np.rad2deg(df[key])

    return df


def get_planets_smass(pdic, num_planets):
    """set up planets class for ttvfast-python

        Args:
            pdic: parameter dataframe from params_for_ttvfast
            num_planets: number of planets

        Returns:
            tuple:
                - list of ttvfast.models.Planet
                - stellar mass (solar unit)

    """
    try:
        import ttvfast
    except ImportError:
        raise ImportError(
            "The ttvfast package https://github.com/mindriot101/ttvfast-python.git is required for this utility function."
        )
    planets = []
    for i in range(num_planets):
        pltag = "%d" % i
        planet_tmp = ttvfast.models.Planet(
            mass=pdic['planet_mass'+pltag],
            period=pdic['period'+pltag],
            eccentricity=pdic['eccentricity'+pltag],
            inclination=pdic['inclination'+pltag],
            longnode=pdic['longnode'+pltag],
            argument=pdic['argument'+pltag],
            mean_anomaly=pdic['mean_anomaly'+pltag]
        )
        planets.append(planet_tmp)

    return planets, float(pdic['star_mass'])


def get_ttvfast_model_rv(pdic, num_planets, t_start, dt, t_end, times_rv, skip_planet_idx=[]):
    """compute transit times using ttvfast-python

        Args:
            pdic: parameter dataframe from params_for_ttvfast
            num_planets: number of planets
            t_start: start time of integration
            dt: integration time step
            t_end: end time of integration
            times_rv: times at which RVs are evaluated

        Returns:
            tuple:
                - list of transit epochs
                - list of transit times
                - array of RVs

    """
    try:
        import ttvfast
    except ImportError:
        raise ImportError(
            "The ttvfast package https://github.com/mindriot101/ttvfast-python.git is required for this utility function."
        )
    planets, smass = get_planets_smass(pdic, num_planets)
    ttvfast_results = ttvfast.ttvfast(
        planets, smass, t_start, dt, t_end, rv_times=list(times_rv))

    idx_planet = np.array(ttvfast_results['positions'][0], 'i')
    transit_epochs = np.array(ttvfast_results['positions'][1], 'i')
    transit_times = np.array(ttvfast_results['positions'][2], 'd')
    rvs = np.array(ttvfast_results['rv'], 'd') * 1.495978707e11 / 86400.

    tnums, tcs = [], []
    for i in range(num_planets):
        if i in skip_planet_idx:
            continue
        idx = (idx_planet == i) & (transit_times > -2)
        tnum, tc = transit_epochs[idx], transit_times[idx]
        tnums.append(tnum)
        tcs.append(tc)

    return tnums, tcs, rvs


def get_ttvfast_model(pdic, num_planets, t_start, dt, t_end, skip_planet_idx=[]):
    """compute transit times using ttvfast-python

        Args:
            pdic: parameter dataframe from params_for_ttvfast
            num_planets: number of planets
            t_start: start time of integration
            dt: integration time step
            t_end: end time of integration

        Returns:
            tuple:
                - list of transit epochs
                - list of transit times

    """
    try:
        import ttvfast
    except ImportError:
        raise ImportError(
            "The ttvfast package https://github.com/mindriot101/ttvfast-python.git is required for this utility function."
        )
    planets, smass = get_planets_smass(pdic, num_planets)
    ttvfast_results = ttvfast.ttvfast(planets, smass, t_start, dt, t_end)

    idx_planet = np.array(ttvfast_results['positions'][0], 'i')
    transit_epochs = np.array(ttvfast_results['positions'][1], 'i')
    transit_times = np.array(ttvfast_results['positions'][2], 'd')

    tnums, tcs = [], []
    for i in range(num_planets):
        if i in skip_planet_idx:
            continue
        idx = (idx_planet == i) & (transit_times > -2)
        tnum, tc = transit_epochs[idx], transit_times[idx]
        tnums.append(tnum)
        tcs.append(tc)

    return tnums, tcs


def get_ttvfast_model_all(pdic, num_planets, t_start, dt, t_end, skip_planet_idx=[]):
    """compute transit times using ttvfast-python

        Args:
            pdic: parameter dataframe from params_for_ttvfast
            num_planets: number of planets
            t_start: start time of integration
            dt: integration time step
            t_end: end time of integration
            skip_planet_idx: list of planet idx to be skipped from output (starting from 0)

        Returns:
            tuple:
                - list of transit epochs
                - list of transit times
                - list of sky-plane distances (au)
                - list of sky-plane velocities (au/day)

    """
    try:
        import ttvfast
    except ImportError:
        raise ImportError(
            "The ttvfast package https://github.com/mindriot101/ttvfast-python.git is required for this utility function."
        )
    planets, smass = get_planets_smass(pdic, num_planets)
    ttvfast_results = ttvfast.ttvfast(planets, smass, t_start, dt, t_end)

    idx_planet = np.array(ttvfast_results['positions'][0], 'i')
    transit_epochs = np.array(ttvfast_results['positions'][1], 'i')
    transit_times = np.array(ttvfast_results['positions'][2], 'd')
    transit_rsky = np.array(ttvfast_results['positions'][3], 'd')
    transit_vsky = np.array(ttvfast_results['positions'][4], 'd')

    tnums, tcs, rskys, vskys = [], [], [], []
    for i in range(num_planets):
        if i in skip_planet_idx:
            continue
        idx = (idx_planet == i) & (transit_times > -2)
        tnum, tc, rsky, vsky = transit_epochs[idx], transit_times[idx], transit_rsky[idx], transit_vsky[idx]
        tnums.append(tnum)
        tcs.append(tc)
        rskys.append(rsky)
        vskys.append(vsky)

    return tnums, tcs, rskys, vskys
