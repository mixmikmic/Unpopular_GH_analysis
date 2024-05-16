from biofloat import ArgoData
from os.path import join, expanduser
ad = ArgoData(cache_file=join(expanduser('~'),
                              'biofloat_fixed_cache_wmo5903264.hdf'),
              bio_list=['DOXY'], variables=('DOXY', 'PSAL', 'TEMP'), verbosity=2)

get_ipython().run_cell_magic('time', '', 'wmo_list = ad.get_cache_file_all_wmo_list()\ndf = ad.get_float_dataframe(wmo_list)\nad.set_verbosity(0)')

df.head()

get_ipython().magic('pylab inline')
import pylab as plt
from mpl_toolkits.basemap import Basemap

def map(lons, lats):
    m = Basemap(llcrnrlon=15, llcrnrlat=-90, urcrnrlon=390, urcrnrlat=90, projection='cyl')
    m.fillcontinents(color='0.8')
    m.scatter(lons, lats, latlon=True, color='red')

map(df.index.get_level_values('lon'), df.index.get_level_values('lat'))

sdf = df.query('pressure < 10').groupby(level=['wmo', 'time', 'lon', 'lat']).mean()
sdf['lon'] = sdf.index.get_level_values('lon')
sdf['lat'] = sdf.index.get_level_values('lat')
sdf['month'] = sdf.index.get_level_values('time').month
sdf['year'] = sdf.index.get_level_values('time').year
sdf['wmo'] = sdf.index.get_level_values('wmo')

msdf = sdf.groupby(['wmo', 'year', 'month']).mean()
from biofloat.utils import o2sat, convert_to_mll
msdf['o2sat'] = 100 * (msdf.DOXY / o2sat(msdf.PSAL, msdf.TEMP))

def round_to(n, increment, mark):
    correction = mark if n >= 0 else -mark
    return int( n / increment) + correction

imsdf = msdf.copy()
imsdf['ilon'] = msdf.apply(lambda x: round_to(x.lon, 1, 0.5), axis=1)
imsdf['ilat'] = msdf.apply(lambda x: round_to(x.lat, 1, 0.5), axis=1)

woa_tmpl = 'http://data.nodc.noaa.gov/thredds/dodsC/woa/WOA13/DATA/o2sat/netcdf/all/1.00/woa13_all_O{:02d}_01.nc'
woa = {}
for m in range(1,13):
    woa[m] = woa_tmpl.format(m)

import xray
def woa_o2sat(month, depth, lon, lat):
    ds = xray.open_dataset(woa[month], decode_times=False)
    return ds.loc[dict(lon=lon, lat=lat, depth=depth)]['O_an'].values[0]

get_ipython().run_cell_magic('time', '', "woadf = imsdf.copy()\nwoadf['month'] = woadf.index.get_level_values('month')\nwoadf['woa_o2sat'] = woadf.apply(lambda x: woa_o2sat(x.month, 5.0, x.ilon, x.ilat), axis=1)")

gdf = woadf[['o2sat', 'woa_o2sat']].copy()
gdf['wmo'] = gdf.index.get_level_values('wmo')
years = gdf.index.get_level_values('year')
months = gdf.index.get_level_values('month')

import matplotlib as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18.0, 4.0)
gdf[['o2sat', 'woa_o2sat']].unstack(level=0).plot()

gdf['gain'] = gdf.woa_o2sat / gdf.o2sat
gdf[['gain']].unstack(level=0).plot()

gdf.groupby('wmo').gain.mean()

