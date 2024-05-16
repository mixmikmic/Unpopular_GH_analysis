get_ipython().magic('matplotlib inline')

from pathlib import Path
from datetime import date

import rasterio
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from scipy import stats
from tqdm import tnrange, tqdm_notebook


import weather_ecmwf
import population_tools

from config import (DATA_SRC, ERA_MONTHLY_FILE, 
                    CLIMATOLOGY_FILE_MONTHLY, POP_DATA_SRC)

era_weather = weather_ecmwf.weather_dataset(ERA_MONTHLY_FILE)
era_weather = era_weather.sel(time=slice('2000','2016'))

era_climatology = weather_ecmwf.climatology_dataset(CLIMATOLOGY_FILE_MONTHLY)

era_climatology.temperature_2m.sel(time='1999-01-01').plot()

clim_t = era_climatology.temperature_2m

def sub_months(era_t_yr):
    """Swap in the time axis so the datasets auto-align
    """
    clim_t['time'] = era_t_yr.time
    return era_t_yr - clim_t
    
delta_t = era_weather.temperature_2m.groupby('time.year').apply(sub_months)

# Climate change in 2000
f = plt.figure(figsize=(6,3))

ax = plt.axes(projection=ccrs.PlateCarree())
delta_t.sel(time='2000').mean(dim='time').plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs=dict(label='K'),
    cmap='RdBu_r',
    vmin=-6, vmax=6
)
ax.coastlines()
ax.set_title('Temperature change in 2000 \n relative to 1986-2008 mean')
plt.tight_layout()

# Climate change in 2016
f = plt.figure(figsize=(6,3))

ax = plt.axes(projection=ccrs.PlateCarree())
delta_t.sel(time='2016').mean(dim='time').plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs=dict(label='K'),
    cmap='RdBu_r',
    vmin=-6, vmax=6
)
ax.coastlines()
ax.set_title('Temperature change in 2016 \n relative to 1986-2008 mean')
plt.tight_layout()
# f.savefig('temperature_change_delta_2000-2016.png')

def collect_summer_delta(delta_t):
    """Wrap in a function to clear temporary vars from memory"""
    # northern hemisphere
    lat_north = delta_t.latitude[delta_t.latitude >= 0]
    lat_south = delta_t.latitude[delta_t.latitude < 0]

    # Summer North
    summer_jja = delta_t['time.season'] == 'JJA'

    # Summer South
    summer_djf = delta_t['time.season'] == 'DJF'

    nh = delta_t.sel(time=summer_jja).groupby('time.year').mean(dim='time')
    sh = delta_t.sel(time=summer_djf).groupby('time.year').mean(dim='time')

    summer_anom = xr.concat([nh.sel(latitude=lat_north),  
                             sh.sel(latitude=lat_south)], dim='latitude')
    return summer_anom

summer_anom = collect_summer_delta(delta_t)

cos_lat = xr.ufuncs.cos(xr.ufuncs.radians(summer_anom.latitude))
summer_anom_ts = (summer_anom.sel(year=slice('2000','2016')) * cos_lat).mean(dim=['latitude', 'longitude'])

summer_anom_ts.plot()
plt.xlabel('Year')
plt.ylabel('Mean warming (K)')

plt.savefig('mean_warming_2000-2016.png', dpi=300)

summer_anom.name = 'summer_global'
summer_anom.to_netcdf(str(DATA_SRC /'lancet' / 'summer_global.nc'))

summer_anom_ts.name = 'summer_global'
summer_anom_ts.to_netcdf(str(DATA_SRC /'lancet' / 'summer_global_ts.nc'))

summer_anom = xr.open_dataarray(str(DATA_SRC /'lancet' / 'summer_global.nc'))

summer_anom_ts = xr.open_dataarray(str(DATA_SRC /'lancet' / 'summer_global_ts.nc'))

def _project_to_population(summer_anom):
    """Wrap in function to control memory use"""
    
    with population_tools.PopulationProjector('population_count_2000-2020.nc') as pop:
        pop_mean = pop.data.mean(dim=['latitude', 'longitude'])
        pop_sum = pop.data.sum(dim=['latitude', 'longitude'])

        def _gen():
            for year in tnrange(2000,2017):
                yield pop.project(year, summer_anom.sel(year=year))
                
        print('Combining')
        summer_exposures = xr.concat(_gen(), dim='year')   
        summer_exposures_ts = (summer_exposures / pop_sum).sum(dim=['latitude', 'longitude']).compute()
        summer_exposures = (summer_exposures / pop_mean).compute()
#         summer_exposures_ts['time'] = summer_exposures_ts['time.year']
        
        return summer_exposures, summer_exposures_ts

summer_exposures, summer_exposures_ts = _project_to_population(summer_anom)

# Plot, rolling the axes
summer_exposures = xr.open_dataarray(str(DATA_SRC /'lancet' / 'summer_exposure.nc'))
newlon = (summer_exposures.longitude - 180).copy()
summer_exposures = summer_exposures.roll(longitude=-len(summer_exposures.longitude)//2)
summer_exposures['longitude'] = newlon
summer_exposures.plot(robust=True,  col='time', col_wrap=4)
plt.savefig('summer_exposures_grid.png')

summer_exposures.name = 'summer_exposure'
summer_exposures.to_netcdf(str(DATA_SRC /'lancet' / 'summer_exposure.nc'))

summer_exposures_ts.name = 'summer_exposure'
summer_exposures_ts.to_netcdf(str(DATA_SRC /'lancet' / 'summer_exposure_ts.nc'))

summer_exposures = xr.open_dataarray(str(DATA_SRC /'lancet' / 'summer_exposure.nc'))
summer_exposures_ts = xr.open_dataarray(str(DATA_SRC /'lancet' / 'summer_exposure_ts.nc'))

summer_exposures_ts.plot()
plt.xlabel('Year')
plt.ylabel('Mean warming exposure (K)')
plt.savefig('mean_warming_experienced_2000-2015.png', dpi=300)
plt.savefig('mean_warming_experienced_2000-2015.pdf', dpi=300)

summer_exposures_ts.plot(color='C0',label='Exposure weighted')
summer_anom_ts.plot(color='C1',label='Area weighted')
# mean_exposures_ts.plot(color='C9', label='Experienced, Year 2000 baseline')

summer_anom_reg = stats.linregress(summer_anom_ts.year.values, summer_anom_ts.values)
summer_anom_reg = (summer_anom_reg.slope * summer_anom_ts.year) + summer_anom_reg.intercept

summer_exposures_reg = stats.linregress(summer_exposures_ts.year.values, summer_exposures_ts.values)
summer_exposures_reg = (summer_exposures_reg.slope * summer_exposures_ts.year) + summer_exposures_reg.intercept

summer_exposures_reg.plot.line('-.', color='C0', label='Exposure trend')
summer_anom_reg.plot.line('--', color='C1', label='Global trend')

plt.xlabel('Year')
plt.ylabel('Mean warming (˚C)')
plt.legend()
plt.savefig('mean_warming_experienced_2000-2016.png', dpi=300)
plt.savefig('mean_warming_experienced_2000-2016.pdf')

output = summer_anom_ts.to_dataframe().join(summer_exposures_ts.to_dataframe())
output.columns = ['Area weighted change K ', 'Exposure weighted change K']

output.to_excel(str(DATA_SRC / 'lancet' / 'temperature_change.xlsx'), sheet_name='temperature_change')

# summer_exposures_ts_rel_2000 = summer_exposures_ts - summer_exposures_ts[0]
# summer_anom_ts_rel_2000 = summer_anom_ts - summer_anom_ts[0]

# summer_exposures_ts_rel_2000.plot(color='C0', label='Exposure weighted')
# summer_anom_ts_rel_2000.plot(color='C1',label='Area weighted')
# # (mean_exposures_ts - mean_exposures_ts[0]).plot(color='C9', label='Year 2000 baseline')

# summer_anom_reg = stats.linregress(summer_anom_ts_rel_2000.year.values, summer_anom_ts_rel_2000.values)
# summer_anom_reg = (summer_anom_reg.slope * summer_anom_ts_rel_2000.year) + summer_anom_reg.intercept

# summer_exposures_reg = stats.linregress(summer_exposures_ts_rel_2000.time.values, summer_exposures_ts_rel_2000.values)
# summer_exposures_reg = (summer_exposures_reg.slope * summer_exposures_ts_rel_2000.time) + summer_exposures_reg.intercept

# summer_exposures_reg.plot.line('-.',color='C0', label='Exposure trend')
# summer_anom_reg.plot.line('--',color='C1', label='Global trend')

# plt.xlabel('Year')
# plt.ylabel('Mean change from 2000 (K)')
# plt.legend()
# plt.savefig('mean_warming_experienced_2000-2015_rel2000.png', dpi=300)

def map_for_year(year):
    year_anom = summer_anom.sel(year=year)

    f = plt.figure(figsize=(6,3))

    ax = plt.axes(projection=ccrs.PlateCarree())
    year_anom.plot.pcolormesh(ax=ax,
                              transform=ccrs.PlateCarree(),
                              cbar_kwargs=dict(label='K'))
    ax.coastlines()
    plt.tight_layout()
    f.savefig(f'temperature_change_{year}.png')

# Plot summer climate change in 2016
f = plt.figure(figsize=(6,3))

ax = plt.axes(projection=ccrs.PlateCarree())
summer_anom.sel(year=2000).plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs=dict(label='K'),
    cmap='RdBu_r',
    vmin=-6, vmax=6)
ax.coastlines()
ax.set_title('Summer temperature difference in 2000 \n relative to 1986-2008 mean')
plt.tight_layout()
f.savefig('summer_temperature_change_2000.png')

# Plot summer climate change in 2016
f = plt.figure(figsize=(6,3))

ax = plt.axes(projection=ccrs.PlateCarree())
summer_anom.sel(year=2016).plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs=dict(label='K'),
    cmap='RdBu_r',
    vmin=-6, vmax=6)
ax.coastlines()
ax.set_title('Summer temperature difference in 2016 \n relative to 1986-2008 mean')
plt.tight_layout()
f.savefig('summer_temperature_change_2016.png')

year_anom_delta = summer_anom.sel(year=2016) - summer_anom.sel(year=2000)

f = plt.figure(figsize=(6,3))

ax = plt.axes(projection=ccrs.PlateCarree())
year_anom_delta.plot.pcolormesh(ax=ax,
                          transform=ccrs.PlateCarree(),
                          cbar_kwargs=dict(label='K'))
ax.coastlines()
ax.set_title('Summer temperature change \n difference between 2000 and 2016')
plt.tight_layout()
f.savefig('temperature_change_delta_2000-2016.png')

year_anom = summer_anom.sel(year=2016)

f = plt.figure(figsize=(6,3))

ax = plt.axes(projection=ccrs.PlateCarree())
year_anom.plot.pcolormesh(ax=ax,
                          transform=ccrs.PlateCarree(),
                          cbar_kwargs=dict(label='K'))
ax.coastlines()
plt.tight_layout()
f.savefig('temperature_change_2016.png')



