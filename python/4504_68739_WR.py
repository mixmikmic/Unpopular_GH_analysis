get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, '../')

from paleopy import proxy
from paleopy import analogs
from paleopy import ensemble

djsons = '../jsons/'
pjsons = '../jsons/proxies'

p = proxy(sitename='Rarotonga',           lon = -159.82,           lat = -21.23,           djsons = djsons,           pjsons = pjsons,           pfname = 'Rarotonga.json',           dataset = 'ersst',           variable ='sst',           measurement ='delta O18',           dating_convention = 'absolute',           calendar = 'gregorian',          chronology = 'historic',           season = 'DJF',           value = 0.6,           calc_anoms = True,           detrend = True)

p.find_analogs()

p.proxy_repr(pprint=True)

from paleopy import WR

w = WR(p, classification='New Zealand')

f = w.plot_bar(sig=1)

f.savefig('/Users/nicolasf/Desktop/proxy.png')

w = WR(p, classification='SW Pacific')

f = w.plot_bar(sig=1)

w.df_probs

ens = ensemble(djsons=djsons, pjsons=pjsons, season='DJF')

classification = 'SW Pacific'

w = WR(ens, classification=classification)

w.parent.description

w.climatology

w.probs_anomalies(kind='many')

w.df_anoms

f = w.plot_heatmap()

f = w.plot_bar()

w.df_anoms.to_csv('/Users/nicolasf/Desktop/table.csv')

w.df_probs_MC



