
__all__ = ["read_testdata_tc"]

import pkg_resources
import glob
import pandas as pd
import numpy as np
from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv.utils import params_to_elements, elements_to_pdic

def read_testdata_tc():
    """read test transit times
    
        Returns:
            initialized JaxTTV class, parameter array, transit time array
    
    """
    path = pkg_resources.resource_filename('jnkepler', 'data/')
    pltag = 'kep51'
    p_init = [45.155305, 85.31646, 130.17809]
    d = pd.read_csv(path+"%s_ttv.txt"%pltag, sep='\s+', header=None, names=['tnum', 'tc', 'tcerr', 'dnum', 'planum'])
    tcobs = [np.array(d.tc[d.planum==j+1]) for j in range(3)]
    errorobs = [np.array(d.tcerr[d.planum==j+1]) for j in range(3)]

    params_test = np.loadtxt(glob.glob(path+"%s*params.txt"%pltag)[0])
    tc_test = np.array(pd.read_csv(glob.glob(path+"%s*tc.csv"%pltag)[0])).ravel()

    dt = 1.0
    t_start, t_end = 155., 2950.

    jttv = JaxTTV(t_start, t_end, dt, tcobs, p_init, errorobs=errorobs, print_info=False)

    pdic = elements_to_pdic(*params_to_elements(params_test, 3))
    
    return jttv, params_test, tc_test, pdic