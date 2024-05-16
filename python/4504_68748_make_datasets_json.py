import json

get_ipython().system('pwd')

dpath = '/Users/nicolasf/CODE/paleopy/data'

datasets = {}


# vcsn: TMean and Rain
datasets['vcsn'] = {}

datasets['vcsn']['TMean'] = {}
datasets['vcsn']['TMean']['path'] = '{}/VCSN_monthly_TMean_1972_2014_grid.nc'.format(dpath)
datasets['vcsn']['TMean']['description'] = 'Mean temperature'
datasets['vcsn']['TMean']['units'] = 'degrees C.'
datasets['vcsn']['TMean']['valid_period'] = (1972, 2014)
datasets['vcsn']['TMean']['domain'] = [166.4, 178.5, -47.4, -34.4]
datasets['vcsn']['TMean']['plot'] = {}
datasets['vcsn']['TMean']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap'


datasets['vcsn']['Rain'] = {}
datasets['vcsn']['Rain']['path'] = '{}/VCSN_monthly_Rain_1972_2014_grid.nc'.format(dpath)
datasets['vcsn']['Rain']['description'] = 'cumulative seasonal precipitation'
datasets['vcsn']['Rain']['units'] = 'mm'
datasets['vcsn']['Rain']['valid_period'] = (1972, 2014)
datasets['vcsn']['Rain']['domain'] = [166.4, 178.5, -47.4, -34.4]
datasets['vcsn']['Rain']['plot'] = {}
datasets['vcsn']['Rain']['plot']['cmap'] = 'palettable.colorbrewer.diverging.BrBG_11.mpl_colormap'

# ersst: sst
datasets['ersst'] = {}
datasets['ersst']['sst'] = {}
datasets['ersst']['sst']['path'] = '{}/ERSST_monthly_SST_1948_2014.nc'.format(dpath)
datasets['ersst']['sst']['description'] = 'Sea Surface Temperature (SST)'
datasets['ersst']['sst']['short_description'] = 'SST'
datasets['ersst']['sst']['units'] = 'degrees C.'
datasets['ersst']['sst']['valid_period'] = (1972, 2014)
datasets['ersst']['sst']['domain'] = [0, 360, -90, 90]
datasets['ersst']['sst']['plot'] = {}
datasets['ersst']['sst']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap'

# gpcp: Rain
datasets['gpcp'] = {}
datasets['gpcp']['Rain'] = {}
datasets['gpcp']['Rain']['path'] = '{}/GPCP_monthly_Rain_1979_2014.nc'.format(dpath)
datasets['gpcp']['Rain']['description'] = 'Average Monthly Rate of Precipitation'
datasets['gpcp']['Rain']['short_description'] = 'GPCP Rain'
datasets['gpcp']['Rain']['units'] = 'mm/day'
datasets['gpcp']['Rain']['valid_period'] = (1979, 2014)
datasets['gpcp']['Rain']['domain'] = [0, 360, -90, 90]
datasets['gpcp']['Rain']['plot'] = {}
datasets['gpcp']['Rain']['plot']['cmap'] = 'palettable.colorbrewer.diverging.BrBG_11.mpl_colormap'


# NCEP: HGT_1000
datasets['ncep'] = {}
datasets['ncep']['hgt_1000'] = {}
datasets['ncep']['hgt_1000']['path'] = '{}/NCEP1_monthly_hgt_1948_2014.nc'.format(dpath)
datasets['ncep']['hgt_1000']['description'] = 'geopotential at 1000 hPa'
datasets['ncep']['hgt_1000']['short_description'] = 'Z1000'
datasets['ncep']['hgt_1000']['units'] = 'meters'
datasets['ncep']['hgt_1000']['valid_period'] = (1948, 2014)
datasets['ncep']['hgt_1000']['domain'] = [0, 360, -90, 90]
datasets['ncep']['hgt_1000']['plot'] = {}
datasets['ncep']['hgt_1000']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap'


# NCEP: HGT_850
datasets['ncep']['hgt_850'] = {}
datasets['ncep']['hgt_850']['path'] = '{}/NCEP1_monthly_hgt_1948_2014.nc'.format(dpath)
datasets['ncep']['hgt_850']['description'] = 'geopotential at 800 hPa'
datasets['ncep']['hgt_850']['short_description'] = 'Z850'
datasets['ncep']['hgt_850']['units'] = 'meters'
datasets['ncep']['hgt_850']['valid_period'] = (1948, 2014)
datasets['ncep']['hgt_850']['domain'] = [0, 360, -90, 90]
datasets['ncep']['hgt_850']['plot'] = {}
datasets['ncep']['hgt_850']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap'

# NCEP: HGT_850
datasets['ncep']['hgt_200'] = {}
datasets['ncep']['hgt_200']['path'] = '{}/NCEP1_monthly_hgt_1948_2014.nc'.format(dpath)
datasets['ncep']['hgt_200']['description'] = 'geopotential at 200 hPa'
datasets['ncep']['hgt_200']['short_description'] = 'Z200'
datasets['ncep']['hgt_200']['units'] = 'meters'
datasets['ncep']['hgt_200']['valid_period'] = (1948, 2014)
datasets['ncep']['hgt_200']['domain'] = [0, 360, -90, 90]
datasets['ncep']['hgt_200']['plot'] = {}
datasets['ncep']['hgt_200']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap'

# NCEP: Omega
datasets['ncep']['omega_500'] = {}
datasets['ncep']['omega_500']['path'] = '{}/NCEP1_monthly_omega_1948_2014.nc'.format(dpath)
datasets['ncep']['omega_500']['description'] = 'Omega at 500 hPa'
datasets['ncep']['omega_500']['short description'] = 'Om. 500'
datasets['ncep']['omega_500']['units'] = 'm/s'
datasets['ncep']['omega_500']['valid_period'] = (1948, 2014)
datasets['ncep']['omega_500']['domain'] = [0, 360, -90, 90]
datasets['ncep']['omega_500']['plot'] = {}
datasets['ncep']['omega_500']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PuOr_11.mpl_colormap'

# NCEP: uwnd 1000
datasets['ncep']['uwnd_1000'] = {}
datasets['ncep']['uwnd_1000']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['uwnd_1000']['description'] = 'zonal wind at 1000 hPa'
datasets['ncep']['uwnd_1000']['short_description'] = 'uwnd1000'
datasets['ncep']['uwnd_1000']['units'] = 'm/s'
datasets['ncep']['uwnd_1000']['valid_period'] = (1948, 2014)
datasets['ncep']['uwnd_1000']['domain'] = [0, 360, -90, 90]
datasets['ncep']['uwnd_1000']['plot'] = {}
datasets['ncep']['uwnd_1000']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: uwnd 850
datasets['ncep']['uwnd_850'] = {}
datasets['ncep']['uwnd_850']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['uwnd_850']['description'] = 'zonal wind at 850 hPa'
datasets['ncep']['uwnd_850']['short_description'] = 'uwnd850'
datasets['ncep']['uwnd_850']['units'] = 'm/s'
datasets['ncep']['uwnd_850']['valid_period'] = (1948, 2014)
datasets['ncep']['uwnd_850']['domain'] = [0, 360, -90, 90]
datasets['ncep']['uwnd_850']['plot'] = {}
datasets['ncep']['uwnd_850']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: uwnd 200
datasets['ncep']['uwnd_200'] = {}
datasets['ncep']['uwnd_200']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['uwnd_200']['description'] = 'zonal wind at 200 hPa'
datasets['ncep']['uwnd_200']['short_description'] = 'unwd200'
datasets['ncep']['uwnd_200']['units'] = 'm/s'
datasets['ncep']['uwnd_200']['valid_period'] = (1948, 2014)
datasets['ncep']['uwnd_200']['domain'] = [0, 360, -90, 90]
datasets['ncep']['uwnd_200']['plot'] = {}
datasets['ncep']['uwnd_200']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: vwnd 1000
datasets['ncep']['vwnd_1000'] = {}
datasets['ncep']['vwnd_1000']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['vwnd_1000']['description'] = 'meridional wind at 1000 hPa'
datasets['ncep']['vwnd_1000']['short_description'] = 'vwnd1000'
datasets['ncep']['vwnd_1000']['units'] = 'm/s'
datasets['ncep']['vwnd_1000']['valid_period'] = (1948, 2014)
datasets['ncep']['vwnd_1000']['domain'] = [0, 360, -90, 90]
datasets['ncep']['vwnd_1000']['plot'] = {}
datasets['ncep']['vwnd_1000']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: vwnd 850
datasets['ncep']['vwnd_850'] = {}
datasets['ncep']['vwnd_850']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['vwnd_850']['description'] = 'meridional wind at 850 hPa'
datasets['ncep']['vwnd_850']['short_description'] = 'vwnd850'
datasets['ncep']['vwnd_850']['units'] = 'm/s'
datasets['ncep']['vwnd_850']['valid_period'] = (1948, 2014)
datasets['ncep']['vwnd_850']['domain'] = [0, 360, -90, 90]
datasets['ncep']['vwnd_850']['plot'] = {}
datasets['ncep']['vwnd_850']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: vwnd 200
datasets['ncep']['vwnd_200'] = {}
datasets['ncep']['vwnd_200']['path'] = '{}/NCEP1_monthly_wind_1948_2014.nc'.format(dpath)
datasets['ncep']['vwnd_200']['description'] = 'meridional wind at 200 hPa'
datasets['ncep']['vwnd_200']['short_description'] = 'vwnd200'
datasets['ncep']['vwnd_200']['units'] = 'm/s'
datasets['ncep']['vwnd_200']['valid_period'] = (1948, 2014)
datasets['ncep']['vwnd_200']['domain'] = [0, 360, -90, 90]
datasets['ncep']['vwnd_200']['plot'] = {}
datasets['ncep']['vwnd_200']['plot']['cmap'] = 'palettable.colorbrewer.diverging.PRGn_11.mpl_colormap'

# NCEP: vwnd 200
datasets['ncep']['Tmean'] = {}
datasets['ncep']['Tmean']['path'] = '{}/NCEP1_monthly_Tmean_1948_2014.nc'.format(dpath)
datasets['ncep']['Tmean']['description'] = 'Mean Temperature at 2m.'
datasets['ncep']['Tmean']['short_description'] = 'T2m'
datasets['ncep']['Tmean']['units'] = 'degrees C.'
datasets['ncep']['Tmean']['valid_period'] = (1948, 2014)
datasets['ncep']['Tmean']['domain'] = [0, 360, -90, 90]
datasets['ncep']['Tmean']['plot'] = {}
datasets['ncep']['Tmean']['plot']['cmap'] = 'palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap'

with open('../../jsons/datasets.json', 'w') as f: 
    json.dump(datasets, f)

d = {}
d['New Zealand'] = {}
d['New Zealand']['Markov Chains'] = '{}/simulations_Kidson_types.hdf5'.format(dpath)
d['New Zealand']['WR_TS'] = '{}/Kidson_Types.csv'.format(dpath)
d['New Zealand']['types'] = ['T', 'SW', 'TNW', 'TSW', 'H', 'HNW', 'W', 'HSE', 'HE', 'NE', 'HW', 'R']
d['New Zealand']['groups'] = {'Trough': ['T', 'SW', 'TNW', 'TSW'], 'Zonal': ['H', 'HNW', 'W'], 'Blocking':['HSE', 'HE', 'NE', 'HW', 'R']}

d['SW Pacific'] = {}
d['SW Pacific']['Markov Chains'] = '{}/simulations_SWPac_types.hdf5'.format(dpath)
d['SW Pacific']['WR_TS'] = '{}/SWPac_Types.csv'.format(dpath)
d['SW Pacific']['types'] = ['SW1', 'SW2', 'SW3', 'SW4', 'SW5', 'SW6']
d['SW Pacific']['groups'] = None

with open('../../jsons/WRs.json', 'w') as f: 
    json.dump(d, f)

d = {}
d['NINO 3.4'] = {}
d['NINO 3.4']['path'] = '{}/NINO34_monthly_1950_2015_1981_2010_Clim.csv'.format(dpath)
d['NINO 3.4']['units'] = 'degrees C.'
d['NINO 3.4']['period'] = (1948, 2014)
d['NINO 3.4']['source'] = 'ERSST NINO3.4 from the Climate Prediction Center'

d['SOI'] = {}
d['SOI']['path'] = '{}/SOI_monthly_1876_2015_1981_2010_Clim.csv'.format(dpath)
d['SOI']['units'] = 'std.'
d['SOI']['period'] = (1948, 2014)
d['SOI']['source'] = 'NIWA SOI'

d['IOD'] = {}
d['IOD']['path'] = '{}/IOD_1900_2014_1981_2010_Clim.csv'.format(dpath)
d['IOD']['units'] = 'std.'
d['IOD']['period'] = (1948, 2014)
d['IOD']['source'] = 'ERSST IOD (NIWA)'

d['SAM'] = {}
d['SAM']['path'] = '{}/SAM_index_1948_2014_1981_2010_Clim.csv'.format(dpath)
d['SAM']['units'] = 'std.'
d['SAM']['period'] = (1948, 2014)
d['SAM']['source'] = 'HGT700 EOF (NIWA)'

with open('../../jsons/indices.json', 'w') as f: 
    json.dump(d, f)

d





