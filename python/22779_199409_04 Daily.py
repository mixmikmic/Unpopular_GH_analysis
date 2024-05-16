get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import StamenTerrain

from pointprocess import *
import pointprocess.plotting as pplot
from lightning_setup import *

city = 'stlouis'
c = Region(city=cities[city])
c.define_grid()

get_ipython().run_cell_magic('time', '', 'top10 = c.get_top(10)')

plt.figure(figsize=(16, 3))
for n in range(1,5):    
    ax = pplot.background(plt.subplot(1, 4, n, projection=ccrs.PlateCarree()))
    ds0 = c.get_daily_ds(top10.index[n-1],func='grid')
    ds0.close()
    c.plot_grid(gaussian_filter(c.FC_grid,2), cmap=cmap, vmin=.5, cbar=True, ax=ax)
    ax.set_title(top10.index[n-1])

c.get_daily_ds('2014-09-26 12:00:00', func='count')

# or equivalently
c.get_daily_ds('2014-09-26 12:00:00', func='grid')
c.FC_grid.sum()

c.get_daily_ds('2014-09-26 12:00:00', func='max')

# or equivalently
c.FC_grid.max()

c.area_over_thresh([1,2,5,10,20])

#location of maximum flash density (lat lon)
yy, xx= np.where(c.FC_grid==c.FC_grid.max())
locmax = c.gridx[xx], c.gridy[yy]

plt.figure(figsize=(10,8))
im, ax = c.plot_grid(gaussian_filter(c.FC_grid, 3), cmap=cmap, vmin=.5, cbar=True, alpha=.7)
ax.add_image(StamenTerrain(), 7)
ax.scatter(locmax[0], locmax[1], c='g', s=50, edgecolor='white', zorder=11)
for lon, lat in zip(locmax[0], locmax[1]):
    ax.text(lon, lat, 'max', zorder=12, horizontalalignment='center',
            verticalalignment='top', fontsize=16);

