from biofloat import ArgoData
ad = ArgoData()

wmo_list = ad.get_oxy_floats_from_status()

sdf, _ = ad._get_df(ad._STATUS)
sdf.ix[:, 'WMO':'GREYLIST'].head()

get_ipython().magic('pylab inline')
def dist_plot(df, title):
    from datetime import date
    ax = df.hist(bins=100)
    ax.set_xlabel('AGE (days)')
    ax.set_ylabel('Count')
    ax.set_title('{} as of {}'.format(title, date.today()))
    
dist_plot(sdf['AGE'], 'Argo float AGE distribution')

sdfq = sdf.query('(AGE != 0) & (OXYGEN == 1) & (GREYLIST != 1)')
dist_plot(sdfq['AGE'], title='Argo oxygen float AGE distribution')
print 'Count age_gte 0340:', len(sdfq.query('AGE >= 340'))
print 'Count age_gte 1000:', len(sdfq.query('AGE >= 1000'))
print 'Count age_gte 2000:', len(sdfq.query('AGE >= 2000'))
print 'Count age_gte 2200:', len(sdfq.query('AGE >= 2200'))
print 'Count age_gte 3000:', len(sdfq.query('AGE >= 3000'))

len(ad.get_oxy_floats_from_status(age_gte=2200))

get_ipython().run_cell_magic('time', '', "from os.path import expanduser, join\nad.set_verbosity(2)\nad = ArgoData(cache_file = join(expanduser('~'), \n     'biofloat_fixed_cache_age2200_profiles2_variablesDOXY_ADJUSTED-PSAL_ADJUSTED-TEMP_ADJUSTED.hdf'))\nwmo_list = ad.get_oxy_floats_from_status(2200)\n# Use 'update_cache=False' to avoid doing lookups for new profile data\ndf = ad.get_float_dataframe(wmo_list, max_profiles=2, update_cache=False)")

# Parameter long_name and units copied from attributes in NetCDF files
time_range = '{} to {}'.format(df.index.get_level_values('time').min(), 
                               df.index.get_level_values('time').max())
parms = {'TEMP_ADJUSTED': 'SEA TEMPERATURE IN SITU ITS-90 SCALE (degree_Celsius)', 
         'PSAL_ADJUSTED': 'PRACTICAL SALINITY (psu)',
         'DOXY_ADJUSTED': 'DISSOLVED OXYGEN (micromole/kg)'}

plt.rcParams['figure.figsize'] = (18.0, 8.0)
fig, ax = plt.subplots(1, len(parms), sharey=True)
ax[0].invert_yaxis()
ax[0].set_ylabel('SEA PRESSURE (decibar)')

for i, (p, label) in enumerate(parms.iteritems()):
    ax[i].set_xlabel(label)
    ax[i].plot(df[p], df.index.get_level_values('pressure'), '.')
    
plt.suptitle('Float(s) ' + ' '.join(wmo_list) + ' from ' + time_range)

import pylab as plt
from mpl_toolkits.basemap import Basemap

plt.rcParams['figure.figsize'] = (18.0, 8.0)
m = Basemap(llcrnrlon=15, llcrnrlat=-90, urcrnrlon=390, urcrnrlat=90, projection='cyl')
m.fillcontinents(color='0.8')

m.scatter(df.index.get_level_values('lon'), df.index.get_level_values('lat'), latlon=True)

