import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# this is where my data path is set
from lightning_setup import out_path

get_ipython().magic('matplotlib inline')

ds = xr.open_mfdataset(out_path+'1993_*_*.nc', concat_dim='record')

x = ds.lon.values
y = ds.lat.values

gridx = np.linspace(x.min(), x.max(), 500)
gridy = np.linspace(y.min(), y.max(), 500)

grid, _, _ = np.histogram2d(x, y, bins=[gridx, gridy])
density = grid.T

# define a good lightning colormap
cmap = mpl.cm.get_cmap('gnuplot_r', 9)
cmap.set_under('None')

#initiate a figure
plt.figure(figsize=(14,5))
ax = plt.axes(projection=ccrs.PlateCarree())

#add some geographic identifying features
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
states = cfeature.NaturalEarthFeature(category='cultural',
                                      name='admin_1_states_provinces_lines',
                                      scale='50m',
                                      facecolor='none')
ax.add_feature(states)
gl = ax.gridlines(draw_labels=True, zorder=4)
gl.xlabels_top = False
gl.ylabels_right = False

# draw the data over top of this template
den = ax.imshow(density, cmap=cmap, interpolation='None', vmin=1,
                extent=[gridx.min(),gridx.max(), gridy.min(), gridy.max()])
plt.title('1993 in accumulated Lightning Flash Count per grid cell', fontsize=16)
plt.colorbar(den, ax=ax)
plt.savefig('./output/US_1993.png')

ds.close()

