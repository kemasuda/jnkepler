
__all__ = ["read_testdata_tc", "read_testdata_4planet"]

import pkg_resources
import glob
import pandas as pd
import numpy as np
from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv.utils import params_to_elements, elements_to_pdic


def read_testdata_tc():
    """read test data

        Returns:
            tuple:
                - JaxTTV class
                - parameter array
                - transit time array 
                - parameter dict

    """
    path = pkg_resources.resource_filename('jnkepler', 'data/')
    pltag = 'kep51'
    p_init = [45.155305, 85.31646, 130.17809]
    d = pd.read_csv(path+"%s_ttv.txt" % pltag, sep='\s+', header=None,
                    names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [np.array(d.tc[d.planum == j+1]) for j in range(3)]
    errorobs = [np.array(d.tcerr[d.planum == j+1]) for j in range(3)]

    params_test = np.loadtxt(glob.glob(path+"%s*params.txt" % pltag)[0])
    tc_test = np.array(pd.read_csv(
        glob.glob(path+"%s*tc.csv" % pltag)[0])).ravel()

    dt = 1.0
    t_start, t_end = 155., 2950.

    jttv = JaxTTV(t_start, t_end, dt, tcobs, p_init,
                  errorobs=errorobs, print_info=False)

    pdic = elements_to_pdic(
        *params_to_elements(params_test, 3), force_coplanar=False)
    pdic['pmass'] = pdic['mass']

    return jttv, params_test, tc_test, pdic


def read_testdata_4planet():
    """read test data (4-planet solution to 3-planet TTVs)

        Returns:
            tuple:
                - JaxTTV class
                - parameter array
                - transit time array 
                - parameter dict

    """
    datapath = pkg_resources.resource_filename(
        'jnkepler', 'data/kep51_4planet.csv')
    parampath = pkg_resources.resource_filename(
        'jnkepler', 'data/kep51_4planet_sol.csv')
    parampath2 = pkg_resources.resource_filename(
        'jnkepler', 'data/kep51_4planet_sol-2.csv')

    d = pd.read_csv(datapath, comment='#')
    tcobs = [np.array(d.tc[d.planet == j]) for j in range(3)]
    errorobs = [np.array(d.tcerr[d.planet == j]) for j in range(3)]
    p_init = [45.155296,  85.316963, 130.175183]
    dt = 1.0
    t_start, t_end = 155., 5600.
    jttv = JaxTTV(t_start, t_end, dt, tcobs, p_init,
                  errorobs=errorobs, print_info=True)

    params = pd.read_csv(parampath)
    params2 = pd.read_csv(parampath2)

    return jttv, params, params2
