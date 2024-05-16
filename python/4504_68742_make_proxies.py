get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
try:
    import xarray as xray 
except: 
    import xray
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, '../')

from paleopy import proxy 
from paleopy import analogs
from paleopy.plotting import scalar_plot

djsons = '../jsons/'
pjsons = '../jsons/proxies'

proxies = pd.read_excel('../data/ProxiesLIANZSWP.xlsx')

proxies.head()

for irow in proxies.index: 
    p = proxy(sitename=proxies.loc[irow,'Site'],           lon = proxies.loc[irow,'Long'],           lat = proxies.loc[irow,'Lat'],           djsons = djsons,           pjsons = pjsons,           pfname = '{}.json'.format(proxies.loc[irow,'Site']),           dataset = proxies.loc[irow,'dataset'],           variable =proxies.loc[irow,'variable'],           measurement ='delta O18',           dating_convention = 'absolute',           calendar = 'gregorian',          chronology = 'historic',           season = 'DJF',           value = proxies.loc[irow,'Anom'],           qualitative = 0,           calc_anoms = 1,           detrend = 1,           method = 'quintiles')
    p.find_analogs()
    p.proxy_repr(pprint=True, outfile=True)



