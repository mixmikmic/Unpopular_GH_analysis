get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import sys

sys.path.append('../')

from paleopy import proxy 
from paleopy import analogs
from paleopy.plotting import indices

djsons = '../jsons/'
pjsons = '../jsons/proxies'

p = proxy(sitename='Rarotonga',           lon = -159.82,           lat = -21.23,           djsons = djsons,           pjsons = pjsons,           pfname = 'Rarotonga.json',           dataset = 'ersst',           variable ='sst',           measurement ='delta O18',           dating_convention = 'absolute',           calendar = 'gregorian',          chronology = 'historic',           season = 'DJF',           value = 0.6,           calc_anoms = 1,           detrend = 1)

p.find_analogs()

p.analog_years

p.analogs

f = p.plot_season_ts()

p.proxy_repr(pprint=True)

indice = indices(p)

indice.composite()

indice.compos.std()

f = indices(p).plot()

from paleopy import ensemble

djsons = '../jsons/'
pjsons = '../jsons/proxies'

ens = ensemble(djsons=djsons, pjsons=pjsons, season='DJF')

f = indices(ens).plot()

obj = indices(p)

obj.composite()

obj.compos



