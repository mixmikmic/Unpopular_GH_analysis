from biofloat import ArgoData
ad = ArgoData(verbosity=2)

get_ipython().run_cell_magic('time', '', "floats340 = ad.get_oxy_floats_from_status(age_gte=340)\nprint('{} floats at least 340 days old'.format(len(floats340)))")

get_ipython().run_cell_magic('time', '', "floats730 = ad.get_oxy_floats_from_status(age_gte=730)\nprint('{} floats at least 730 days old'.format(len(floats730)))")

get_ipython().run_cell_magic('time', '', 'dac_urls = ad.get_dac_urls(floats340)\nprint(len(dac_urls))')

get_ipython().run_cell_magic('time', '', "wmo_list = ['1900650']\nad.set_verbosity(0)\ndf = ad.get_float_dataframe(wmo_list, max_profiles=20)")

get_ipython().run_cell_magic('time', '', 'df = ad.get_float_dataframe(wmo_list, max_profiles=20)')

df.head()

time_range = '{} to {}'.format(df.index.get_level_values('time').min(), 
                               df.index.get_level_values('time').max())
df.query('pressure < 10')

df.query('pressure < 10').groupby(level=['wmo', 'time']).mean()

get_ipython().magic('pylab inline')
import pylab as plt
# Parameter long_name and units copied from attributes in NetCDF files
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

from mpl_toolkits.basemap import Basemap

m = Basemap(llcrnrlon=15, llcrnrlat=-90, urcrnrlon=390, urcrnrlat=90, projection='cyl')
m.fillcontinents(color='0.8')

m.scatter(df.index.get_level_values('lon'), df.index.get_level_values('lat'), latlon=True)
plt.title('Float(s) ' + ' '.join(wmo_list) + ' from ' + time_range)

