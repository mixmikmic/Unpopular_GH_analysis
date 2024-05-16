get_ipython().magic('matplotlib inline')

from pathlib import Path
from datetime import date

import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy
from cartopy import crs

import geopandas as gpd 
from config import DATA_SRC,POP_DATA_SRC, ERA_MONTHLY_FILE, CLIMATOLOGY_FILE_RESAMP, SHAPEFILES_SRC
import weather_ecmwf
import util
import config

import climate_conversions
import population_tools

spi_file = DATA_SRC / 'weather' / 'spi3_6_12_1deg_cru_ts_3_21_1949_2012.nc'

spi = xr.open_dataset(str(spi_file))

spi

spi.spi3.mean(dim=['lat','lon']).plot()

droughts = spi.spi3.where(spi.spi3 < -1.28)

droughts = droughts.groupby('time.year').count(dim='time')
droughts = droughts.astype(np.float64)
droughts.name = 'n_severe_drought'

droughts.sum(dim=['lat', 'lon']).plot()

droughts.sel(year=2003).plot()

def get_drought_projection(droughts):
    with population_tools.PopulationProjector() as pop:
        years = list(range(2000,2013))
        def _gen():
            for year in years:
                print(year)
                yield pop.project(year, droughts.sel(year=year))
        droughts_pop = xr.concat(_gen(), dim='time')
        pop_sum = pop.data.population.sum(dim=['latitude', 'longitude'])
        return droughts_pop.compute(), pop_sum.compute()
    
droughts_pop, pop_sum = get_drought_projection(droughts)

droughts_pop_ts = (droughts_pop / pop_sum).sum(dim=['latitude', 'longitude'])

droughts_pop_ts.plot()



