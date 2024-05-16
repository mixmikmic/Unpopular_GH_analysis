import os
import numpy as np
import pandas as pd
import xarray as xr

from pointprocess import *
from lightning_setup import *
get_ipython().magic('matplotlib inline')

c = Region(city=cities['cedar'])
c.SUBSETTED = False
c.CENTER = (37.7, -111.8)
c.RADIUS = 0.7

storm = '2010-07-20'

ds = c.get_daily_ds(storm, filter_CG=dict(method='CG'), func=None)

c.conditional_rate_of_occurrence(ds);

foo = pd.DataFrame(np.random.rand(10000, 2), columns=['speed', 'direction'])
foo['direction'] *= 360

from windrose import WindroseAxes
ax = WindroseAxes.from_ax()
ax.bar(foo['direction'], foo['speed'], bins=[0, .2, .4, .6, .8], normed=True, opening=0.9, edgecolor='white')
ax.set_legend()



