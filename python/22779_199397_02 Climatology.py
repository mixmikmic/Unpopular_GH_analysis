import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import cartopy.crs as ccrs
from cartopy.io.img_tiles import *

from scipy.ndimage.filters import gaussian_filter

get_ipython().magic('matplotlib inline')

from pointprocess import *
import pointprocess.plotting as pplot
from lightning_setup import *

SR_LOC = os.environ.get('SR_LOC')

c = Region(city=cities['cedar'])
c.define_grid(nbins=600)
c.CENTER

get_ipython().run_cell_magic('time', '', 'MMDC_grid = {}\nMMFC_grid = {}\nfor m in range(1,13):\n    ds = c.get_ds(m=m)\n    print(m)\n    c.to_DC_grid(ds)\n    ds.close()\n    MMDC_grid.update({m: c.DC_grid})\n    MMFC_grid.update({m: c.FC_grid})')

get_ipython().run_cell_magic('time', '', 'MMFC = np.stack(MMFC_grid.values(), axis=0)\nFC = np.sum(MMFC, axis=(0))\n\n#annually averaged rate of occurrence of lightning\nmean_annual_FD = FC/float(2016-1991)\nsmoothed = gaussian_filter(mean_annual_FD, 2)')

plt.figure(figsize=(10,8))
im, ax = c.plot_grid(smoothed, cmap=cmap, cbar=True, zorder=5.5, vmin=1, vmax=5, alpha=.7)
ax.add_image(pplot.ShadedReliefESRI(), 6)
pplot.urban(ax, edgecolor='white', linewidth=2)

plt.figure(figsize=(16, 3))
n=1
for m in [6,7,8,9]:
    ax = pplot.background(plt.subplot(1, 4, n, projection=ccrs.PlateCarree()))
    c.plot_grid(MMFC_grid[m], cmap=cmap, vmin=1, vmax=50, cbar=True, ax=ax, zorder=5)
    ax.set_title(months[m])
    n+=1

JAFC_grid = MMFC_grid[7]+MMFC_grid[8]
img = gaussian_filter(JAFC_grid, 4)

plt.figure(figsize=(16,6))

ax = pplot.background(plt.subplot(1, 2, 1, projection=ccrs.PlateCarree()))
im, ax = c.plot_grid(img, cmap=cmap, cbar=True, zorder=5, alpha=.7, vmin=1, ax=ax)
ax.add_image(pplot.ShadedReliefESRI(), 6)
ax.set_extent([c.gridx.min(), c.gridx.max(), c.gridy.min(), c.gridy.max()])
ax.set_title('July August Accumulated Flash Count')

ax = pplot.background(plt.subplot(1, 2, 2,projection=ccrs.PlateCarree()))
ax.set_extent([c.gridx.min(), c.gridx.max(), c.gridy.min(), c.gridy.max()])
ax.add_image(StamenTerrain(), 7)
ax.contour(c.gridx[:-1], c.gridy[:-1], img, cmap=cmap, zorder=5, linewidths=3)
ax.set_title('July August Accumulated Flash Count')

#plt.suptitle('Comparisons of different smoothed ways of viewing the total Flash Counts', fontsize=18);

import matplotlib.cm as cm

MMDC = pd.DataFrame(np.array([[np.sum(MMDC_grid[m][hr]) for hr in range(0,24)] for m in months.keys()]).T)
MMDC.columns = range(1,13)

MMDC.loc[24, :] = MMDC.loc[0,:]
MMDC.index=(MMDC.index/24)*2*np.pi
log_MMDC = np.log10(MMDC)

plt.figure(figsize=(8, 8))
for m in months.keys():
    ax = plt.subplot(1,1,1, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    log_MMDC.plot(color=cm.hsv(np.linspace(0, 1, 12)), ax=ax, title='Monthly diurnal cycle for flash count \n')
    plt.legend(months.values(), loc=(.9, 0), fontsize=10)
    ticks = np.linspace(0, 2*np.pi, 9)[1:]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['{num}:00'.format(num=int(theta/(2*np.pi)*24)) for theta in ticks])
    ax.set_rticks(range(1,7))
    ax.set_yticklabels(['{x} strikes'.format(x=10**x) for x in range(1,7)])
    ax.grid()

JADC_grid = {}
for k,v in MMDC_grid[7].items():
    JADC_grid.update({k: v + MMDC_grid[8][k]})

JADC_gauss = {}
for k,v in JADC_grid.items():
    JADC_gauss.update({k:gaussian_filter(v, 4)})

plt.figure(figsize=(16,3))
n=1
step = 2
for i in range(8,16,step):
    q=np.zeros(JADC_grid[0].shape)
    for hr in h[i:i+step]:
        q+=JADC_grid[hr]
    ax = pplot.background(plt.subplot(1, 4, n, projection=ccrs.PlateCarree()))
    ax.set_title('{step} hour starting at {t:02d}:00 UTC'.format(step=step, t=h[i]))
    c.plot_grid(q,cmap=cmap, vmin=1, vmax=45, cbar=True, ax=ax, zorder=10)
    n+=1

plt.figure(figsize=(16,6))

cmap_husl = mpl.colors.ListedColormap(sns.husl_palette(256, .2, l=.6, s=1))
img = np.argmax(np.stack(JADC_gauss.values()), axis=0)

ax = pplot.background(plt.subplot(1, 2, 2, projection=ccrs.PlateCarree()))
im, ax = c.plot_grid(img, cmap=cmap_husl, vmin=0, vmax=24, cbar=True, alpha=.8, ax=ax, zorder=5)
ax.add_image(pplot.ShadedReliefESRI(), 6)
ax.set_extent([c.gridx.min(), c.gridx.max(), c.gridy.min(), c.gridy.max()])
ax.set_title('July August hour of peak FC');

plt.savefig('output/cedar/JA hour of peak FC.png')

plt.figure(figsize=(16, 9))
for m in months.keys():
    ax = pplot.background(plt.subplot(3, 4, m, projection=ccrs.PlateCarree()))
    hourly3D = np.stack(MMDC_grid[m].values())
    amplitude = ((np.max(hourly3D, axis=0)-np.min(hourly3D, axis=0))/np.mean(hourly3D, axis=0))
    amplitude = np.nan_to_num(amplitude)
    c.plot_grid(amplitude, cmap=cmap, cbar=True, vmin=.0001, ax=ax)
    ax.set_title(months[m])

plt.figure(figsize=(16,6))

hourly3D = np.stack(np.stack(JADC_gauss.values()))
amplitude = ((np.max(hourly3D, axis=0)-np.min(hourly3D, axis=0))/np.mean(hourly3D, axis=0))
amplitude = np.nan_to_num(amplitude)
ax = pplot.background(plt.subplot(1, 2, 1, projection=ccrs.PlateCarree()))
im, ax = c.plot_grid(amplitude, cmap=cmap, cbar=True, ax=ax, zorder=5, alpha=.7)
ax.add_image(pplot.ShadedReliefESRI(), 6)
ax.set_extent([c.gridx.min(), c.gridx.max(), c.gridy.min(), c.gridy.max()])
ax.set_title('July August amplitude of DC');

top10 = c.get_top(10)
top10

ds = xr.open_mfdataset(c.PATH+'2012_*_*.nc')
df = ds.to_dataframe()
ds.close()

plt.figure(figsize=(8,4))
plt.hist([df[df['cloud_ground'] == b'C']['amplitude'],df[df['cloud_ground'] == b'G']['amplitude']], 
         bins=10, range=(-50, 50))
plt.ylabel('Flash Count')
plt.title('Cedar City area 2012 amplitude of flash count by type')
plt.xlabel('Amplitude (kA)')
plt.legend(labels=['cloud to cloud', 'cloud to ground']);



