
__all__ = ["params_for_ttvfast", "get_ttvfast_model"]

import pandas as pd
from jax import jit
from .utils import convert_elements

def params_for_ttvfast(samples, t_epoch, WHsplit=True, angles_in_degrees=True,
                       names=["period", "eccentricity", "inclination", "argument", "longnode", "mean_anomaly"]):
    """ convert JaxTTV samples into TTVFast (or other) format

        Args:
            samples: mcmc.get_samples()
            t_epoch: time at which osculating elements are defined
            WHsplit: True for TTVFast
            angles_in_degrees: If True, angles are in degrees

        Returns:
            dataframe containing parameters

    """
    def func(e, m):
        return convert_elements(e, m, t_epoch, WHsplit=WHsplit)
    convert_elements_map = jit(vmap)(func, (0,0,), 0)
    elements, masses = convert_elements_map(samples['elements'], samples['masses'])

    pdic = {}
    for j in range(num_planets):
        pdic['planet_mass%d'%j] = masses[:,j+1]
        for i,n in enumerate(names):
            pdic[n+"%d"%j] = elements[:,i+1,j]
    pdic['star_mass'] = masses[:, 0]
    pdic['num_planets'] = num_planets
    df = pd.DataFrame(data=pdic)

    if angles_in_degrees:
        for key in df.keys():
            if "inclination" in key or "argument" in key or "node" in key or "anomaly" in key:
                df[key] = np.rad2deg(df[key])

    return df


def get_planets_smass(pdic, num_planets):
    """ set up planets class for ttvfast-python

        Args:
            pdic: parameter dataframe from params_for_ttvfast
            num_planets: number of planets

        Returns:
            list of ttvfast.models.Planet
            stellar mass (solar unit)

    """
    import ttvfast
    planets = []
    for i in range(num_planets):
        pltag = "%d"%i
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

    return planets, float(pdic.star_mass)


def get_ttvfast_model(pdic, num_planets, t_start, dt, t_end):
    """ compute transit times using ttvfast-python

        Args:
            pdic: parameter dataframe from params_for_ttvfast
            num_planets: number of planets
            t_start: start time of integration
            dt: integration time step
            t_end: end time of integration

        Returns:
            list of transit epochs
            list of transit times

    """
    import ttvfast
    planets, smass = get_planets_smass(pdic, num_planets)
    ttvfast_results = ttvfast.ttvfast(planets, smass, t_start, dt, t_end)

    idx_planet = np.array(ttvfast_results['positions'][0],'i')
    transit_epochs = np.array(ttvfast_results['positions'][1],'i')
    transit_times = np.array(ttvfast_results['positions'][2],'d')

    tnums, tcs = [], []
    for i in range(num_planets):
        idx = (idx_planet == i) & (transit_times > -2)
        tnum, tc = transit_epochs[idx], transit_times[idx]
        tnums.append(tnum)
        tcs.append(tc)

    return tnums, tcs
