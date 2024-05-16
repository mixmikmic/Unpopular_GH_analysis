import numpy as np
import pandas as pd
import xarray as xr

from pointprocess import *
from lightning_setup import *

city = 'cedar'
c = Region(city=cities[city])
c.define_grid(step=1, units='km')

get_ipython().run_cell_magic('time', '', "def get_FC(region, y):\n    # filter CG using the flag in recent data, or by taking only strikes with\n    # amblitude less than a value, or less than one value and greater than another\n    ds = region.get_ds(y=y, filter_CG=dict(method='less_than', amax=-10), func=None)\n    df = ds.to_dataframe()\n    ds.close()\n    df.index = df.time\n    FC = df['lat'].resample('24H', base=12, label='right').count()\n    FC.name = 'FC'\n    return FC\n\nFC = get_FC(c, 1996)\nfor y in range(1997,2017):\n    FC = pd.concat([FC, get_FC(c, y)])")

FC.sort_values(ascending=False, inplace=True)
FC.head(10)

top_50 = FC.head(50)
top_200 = FC.head(200)
top_200.to_csv("Cedar_top_200_days_2010_2016.csv")

EPK = Region(city=cities['cedar'])
EPK.SUBSETTED = False
EPK.CENTER = (37.7, -111.8)
EPK.RADIUS = 0.7
EPK.define_grid(step=1, units='km')

get_ipython().run_cell_magic('time', '', 'FC = get_FC(EPK, 1996)\nfor y in range(1997,2017):\n    FC = pd.concat([FC, get_FC(EPK, y)])')

FC.sort_values(ascending=False, inplace=True)
FC.head(10)

EPK.ngrid_cells = (EPK.gridx.size-1) * (EPK.gridy.size-1)

flash_density = FC/float(EPK.ngrid_cells)

