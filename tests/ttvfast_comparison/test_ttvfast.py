#%%
import pytest
import ttvfast
import glob
import pandas as pd
import numpy as np
from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv.utils import params_to_elements
import itertools
import timeit
import pkg_resources
path = pkg_resources.resource_filename('jnkepler', 'data/')

#%%
def params_for_ttvfast(pdic, stellar_mass=1.0):
    # list of planet models
    planets = []
    npl = int(pdic.n_pl)
    for i in range(npl):
        pltag = "%d"%i
        planet_tmp = ttvfast.models.Planet(
            mass=pdic['pmass'+pltag],
            period=pdic['period'+pltag],
            eccentricity=pdic['e'+pltag],
            inclination=pdic['incl'+pltag],
            longnode=pdic['lnode'+pltag],
            argument=pdic['omega'+pltag],
            mean_anomaly=pdic['M'+pltag]
        )
        planets.append(planet_tmp)

    return planets, stellar_mass, float(pdic['t_start']), float(pdic['dt']), float(pdic['t_end']), npl

def init_jaxttv(ttvfast_results, t_start, t_end, dt, npl):
    idx_planet = np.array(ttvfast_results['positions'][0],'i')
    transit_epochs = np.array(ttvfast_results['positions'][1],'i')
    transit_times = np.array(ttvfast_results['positions'][2],'d')

    tcobs, errorobs, p_init = [], [], []
    for i in range(npl):
        idx = (idx_planet==i) & (transit_times>-2)
        tnum, tc = transit_epochs[idx], transit_times[idx]
        tcobs.append(tc)
        p, t0 = np.polyfit(tnum, tc, deg=1)
        p_init.append(p)
    p_init = np.array(p_init)

    jttv = JaxTTV(t_start, t_end, dt)
    jttv.set_tcobs(tcobs, p_init)

    return jttv, np.array(list(itertools.chain.from_iterable(tcobs)))

def compare_transit_times(pdic_ttvfast, params_jttv, time=False, dt_factor=1.):
    planets, smass, t_start, dt, t_end, npl = params_for_ttvfast(pdic_ttvfast)
    ttvfast_results = ttvfast.ttvfast(planets, smass, t_start, dt, t_end)
    jttv, tc_ttvfast = init_jaxttv(ttvfast_results, t_start, t_end, dt*dt_factor, npl)
    tc_jttv, de = jttv.get_ttvs(*params_to_elements(params_jttv, jttv.nplanet))

    if time:
        names = {**globals(), **locals()}
        loop = 100
        result = timeit.timeit('ttvfast.ttvfast(planets, smass, t_start, dt, t_end)', globals=names, number=loop)
        result_str = "%.2f ms per loop (%d loops)"%(result / loop * 1000, loop)
        print ()
        print ("TTVFast timeit:", result_str)
        result = timeit.timeit('jttv.get_ttvs(*params_to_elements(params_jttv, jttv.nplanet))', globals=names, number=loop)
        result_str = "%.2f ms per loop (%d loops)"%(result / loop * 1000, loop)
        print ("JaxTTV timeit:", result_str)

    return tc_jttv, tc_ttvfast

def test_comparison():
    pltag = 'kep51'
    pdic_ttvfast = pd.read_csv(glob.glob(path+"%s*pdict_ttvfast.csv"%pltag)[0])
    params_jttv = np.loadtxt(glob.glob(path+"%s*params.txt"%pltag)[0])
    tc_jttv, tc_ttvfast = compare_transit_times(pdic_ttvfast, params_jttv, time=False, dt_factor=1.)
    tc_difference = np.array(tc_jttv - tc_ttvfast)
    diff_max_sec = np.max(np.abs(tc_difference)*86400.)
    print ("max difference from TTVFast (sec):", diff_max_sec)
    assert diff_max_sec < 1.0

#%%
if __name__ == '__main__':
    test_comparison()
