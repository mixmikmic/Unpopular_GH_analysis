get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pointprocess import *
import pointprocess.plotting as pplot
from lightning_setup import *
import cartopy.crs as ccrs
from cartopy.io.img_tiles import StamenTerrain

c = Region(city=cities['cedar'])
c.define_grid()

# initialize some variable names
df=None
df_10=None
df_20=None
df_50=None

get_ipython().run_cell_magic('time', '', "# open dataset for the months that you are interested in\nds = c.get_ds(m=7, filter_CG=dict(method='less_than', amax=-10), func=None)\n\n# get the time difference between each strike\nt_diff = pd.TimedeltaIndex(np.diff(ds.time))\n\n# bool of whether difference is greater than 1 hour and less than 20 days (aka not a whole year)\ncond=(t_diff>pd.Timedelta(hours=1)) & (t_diff<pd.Timedelta(days=20))\n\n# make a dataframe of all the records when this is the case and reset the index\ndf0 = ds.record[1:][cond].to_dataframe().drop('record', axis=1).reset_index()\n\n# count the strikes in the next hour\nml = [((ds.time>t.asm8) & (ds.time<(t+ pd.Timedelta(hours=1)).asm8)).sum().values for t in df0.time]\n\nds.close()")

df = pd.concat([df, df0])

df_10 = pd.concat([df_10, df0[np.array(ml)>10]])
df_20 = pd.concat([df_20, df0[np.array(ml)>20]])
df_50 = pd.concat([df_50, df0[np.array(ml)>50]])

plt.figure(figsize=(8,8))
ax = plt.axes(projection=ccrs.PlateCarree())
pplot.background(ax)
kwargs=dict(x='lon', y='lat', ax=ax, s=50, edgecolor='None')
df.plot.scatter(c='r', **kwargs)
df_10.plot.scatter(c='y', **kwargs)
df_20.plot.scatter(c='g', **kwargs)
df_50.plot.scatter(c='b', **kwargs)
ax.set_extent([c.gridx.min()+1, c.gridx.max()-1, c.gridy.min()+1, c.gridy.max()-1])
ax.add_image(StamenTerrain(), 7)
plt.legend(['Strikes in', 'next hour',
            '<10', '>10','>20', '>50'], loc='center left');
plt.savefig('./output/cedar/JA Initiation Locations.png')

