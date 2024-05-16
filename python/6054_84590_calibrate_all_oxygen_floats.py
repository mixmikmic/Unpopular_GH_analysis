from biofloat import ArgoData
from os.path import join, expanduser
ad = ArgoData(cache_file = join(expanduser('~'), 
     'biofloat_fixed_cache_age365_variablesDOXY_ADJUSTED-PSAL_ADJUSTED-TEMP_ADJUSTED.hdf'))

ocdf = ad.get_cache_file_oxy_count_df()
print ocdf.groupby('wmo').sum().sum()
print 'Float having DOXY_ADJUSTED data:', ocdf.wmo.count()
acdf = ad.get_cache_file_all_wmo_list()
print 'Number of floats examined:', len(acdf)

import pandas as pd
df = pd.DataFrame()
with pd.HDFStore(join(expanduser('~'), 'woa_lookup_age365.hdf')) as s:
    wmo_list = ocdf.wmo
    for wmo in wmo_list:
        try:
            fdf = s.get(('/WOA_WMO_{}').format(wmo))
        except KeyError:
            pass
        if not fdf.dropna().empty:
            df = df.append(fdf)

print df.head()
print df.describe()
print 'Number of floats with corresponding WOA data:', len(df.index.get_level_values('wmo').unique())

qdf = df.query('(o2sat > 50 ) & (o2sat < 200)')
qdf.describe()

get_ipython().magic('pylab inline')
import pylab as plt
plt.rcParams['figure.figsize'] = (18.0, 4.0)
plt.style.use('ggplot')
ax = qdf.groupby('wmo').mean().gain.hist(bins=100)
ax.set_xlabel('Gain')
ax.set_ylabel('Count')
floats = qdf.index.get_level_values('wmo').unique()
ax.set_title(('Distribution of WOA calibrated gains from {} floats').format(len(floats)))

qdf.head()

plt.rcParams['figure.figsize'] = (18.0, 8.0)
ax = qdf.unstack(level='wmo').gain.plot()
ax.set_ylabel('Gain')
ax.set_title(('Calculated gain factor for {} floats').format(len(floats)))
ax.legend_.remove()

wmo_list = qdf.index.get_level_values('wmo').unique()
colors = cm.spectral(np.linspace(0, 1, len(wmo_list)))
print 'Number of floats with reasonable oxygen saturation values:', len(wmo_list)

plt.rcParams['figure.figsize'] = (18.0, 8.0)
fig, ax = plt.subplots(1, 1)
for wmo, c in zip(wmo_list, colors):
    ax.scatter(qdf.xs(wmo, level='wmo')['o2sat'], qdf.xs(wmo, level='wmo')['woa_o2sat'], c=c)
ax.set_xlim([40, 200])
ax.set_ylim([40, 200])
ax.set_xlabel('Float o2sat (%)')
ax.set_ylabel('WOA o2sat (%)')

get_ipython().run_cell_magic('time', '', 'ad.set_verbosity(0)\ndf1 = ad.get_float_dataframe(wmo_list, update_cache=False, max_profiles=4)')

from mpl_toolkits.basemap import Basemap
m = Basemap(llcrnrlon=15, llcrnrlat=-90, urcrnrlon=390, urcrnrlat=90, projection='cyl')
m.fillcontinents(color='0.8')
df1m = df1.groupby(level=['wmo','lon','lat']).mean()
for wmo, c in zip(wmo_list, colors):
    try:
        lons = df1m.xs(wmo, level='wmo').index.get_level_values('lon')
        lats = df1m.xs(wmo, level='wmo').index.get_level_values('lat')
        try:
            m.scatter(lons, lats, latlon=True, color=c)
        except IndexError:
            # Some floats have too few points
            pass
        lon, lat = lons[0], lats[0]
        if lon < 0:
            lon += 360
        plt.text(lon, lat, wmo)
    except KeyError:
        pass



