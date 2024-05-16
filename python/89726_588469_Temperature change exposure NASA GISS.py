get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from affine import Affine
from tqdm import tnrange, tqdm_notebook


import population_tools
from config import DATA_SRC, POP_DATA_SRC

nasa_giss_anom = DATA_SRC / 'weather' / 'nasa_giss' / 'air.2x2.1200.mon.anom.comb.nc'

nasa_giss = xr.open_dataset(str(nasa_giss_anom))
nasa_giss = nasa_giss.rename({'lon':'longitude', 'lat':'latitude'})
nasa_giss

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

summer_anom = collect_summer_delta(nasa_giss)

nasa_giss.mean(dim=['latitude', 'longitude']).air.plot()

summer_anom.mean(dim=['latitude', 'longitude']).air.plot()

# temperature = nasa_giss.air.groupby('time.year').mean(dim='time')

target = xr.open_dataset(str(POP_DATA_SRC / 'histsoc_population_0.5deg_1861-2005.nc'),                         )

with population_tools.PopulationProjector('histsoc_population_0.5deg_1861-2005_.nc', mask_empty=False) as pop:
    da = xr.DataArray(pop.water_mask, coords=[pop.data.latitude, pop.data.longitude])
    da.plot()

with population_tools.PopulationProjector('histsoc_population_0.5deg_1861-2005.nc') as pop:
    pop_mean = pop.data.mean(dim=['latitude', 'longitude'])
    pop_sum = pop.data.sum(dim=['latitude', 'longitude'])


    def _gen():
        for year in tnrange(2000,2017):
            yield pop.project(year, summer_anom.sel(year=year))

    summer_exposures = xr.concat(_gen(), dim='year')   

yr_exposures[0].plot(robust=True)

yr_exposures = xr.concat(yr_exposures, dim='time')
yr_exposures.name = 'year_mean_exposure'

yr_exposures.to_netcdf(str(DATA_SRC / 'lancet' / 'nasa_giss_yr_exposure.nc'))



