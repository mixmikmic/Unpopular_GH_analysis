# this is a list of packages that are used in this notebook
# these come with python
import io
import zipfile
import functools
import bisect
import datetime

# you can install these packages using pip or anaconda
# (requests numpy pandas bokeh pyproj statsmodels)

# for downloading
import requests
import netCDF4

# computation libraries
import numpy as np
import pandas

# coordinate systems
import pyproj

# statistics
import statsmodels.api as sm
import statsmodels

# plotting
import bokeh.charts
import bokeh.io
import bokeh.plotting
import bokeh.tile_providers
import bokeh.palettes

import windrose
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
# matplotlib.projections.register_projection(windrose.WindroseAxes)
# print(matplotlib.projections.get_projection_names())
import cmocean.cm

# displaying things
from ipywidgets import Image
import IPython.display

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def find_closest(lat, lon, lat_i, lon_i):
    """lookup the index of the closest lat/lon"""
    Lon, Lat = np.meshgrid(lon, lat)
    idx = np.argmin(((Lat - lat_i)**2 + (Lon - lon_i)**2))
    Lat.ravel()[idx], Lon.ravel()[idx]
    [i, j] = np.unravel_index(idx, Lat.shape)
    return i, j


def make_wind_df(lat_i, lon_i):
    """create a dataset for wind, for 1 latitude/longitude"""
    u_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/uwnd.10m.mon.mean.nc'
    v_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/vwnd.10m.mon.mean.nc'

    # open the 2 files
    ds_u = netCDF4.Dataset(u_file)
    ds_v = netCDF4.Dataset(v_file)

    # read lat,lon, time from 1 dataset
    lat = ds_u.variables['lat'][:]
    lon = ds_u.variables['lon'][:]
    time = ds_u.variables['time'][:]
    # check with the others
    lat_v = ds_v.variables['lat'][:]
    lon_v = ds_v.variables['lon'][:]
    time_v = ds_v.variables['time'][:]
    assert (lat == lat_v).all() and (lon == lon_v).all() and (time == time_v).all()
    # convert to datetime
    t = netCDF4.num2date(time, ds_u.variables['time'].units)

    # this is the index where we want our data
    i, j = find_closest(lat, lon, lat_i, lon_i)
    # get the u, v variables
    # print('found point', lat[i], lon[j])
    u = ds_u.variables['uwnd'][:, i, j]
    v = ds_v.variables['vwnd'][:, i, j]
    # compute derived quantities
    speed = np.sqrt(u ** 2 + v ** 2)
    # compute direction in 0-2pi domain
    direction = np.mod(np.angle(u + v * 1j), 2 * np.pi)
    # put everything in a dataframe
    wind_df = pandas.DataFrame(data=dict(u=u, v=v, t=t,
                                         speed=speed, direction=direction))
    # return it
    return wind_df


lat_i = 53
lon_i = 3
wind_df = make_wind_df(lat_i=lat_i, lon_i=lon_i)

urls = {
    'metric_monthly':
    'http://www.psmsl.org/data/obtaining/met.monthly.data/met_monthly.zip',
    'rlr_monthly':
    'http://www.psmsl.org/data/obtaining/rlr.monthly.data/rlr_monthly.zip',
    'rlr_annual':
    'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip'
}
dataset_name = 'rlr_monthly'

# these compute the rlr back to NAP
# lambda functions are not recommended by PEP8, 
# but not sure how to replace them
main_stations = {
    20: {
        'name': 'Vlissingen',
        'rlr2nap': lambda x: x - (6976-46)
    },
    22: {
        'name': 'Hoek van Holland',
        'rlr2nap': lambda x: x - (6994 - 121)
    },
    23: {
        'name': 'Den Helder',
        'rlr2nap': lambda x: x - (6988-42)
    },
    24: {
        'name': 'Delfzijl',
        'rlr2nap': lambda x: x - (6978-155)
    },
    25: {
        'name': 'Harlingen',
        'rlr2nap': lambda x: x - (7036-122)
    },
    32: {
        'name': 'IJmuiden',
        'rlr2nap': lambda x: x - (7033-83)
    },
#     1551: {
#         'name': 'Roompot buiten',
#         'rlr2nap': lambda x: x - (7011-17)
#     },
#     9: {
#         'name': 'Maassluis',
#         'rlr2nap': lambda x: x - (6983-184)
#     },
#     236: {
#         'name': 'West-Terschelling',
#         'rlr2nap': lambda x: x - (7011-54)
#     }
}

# the main stations are defined by their ids
main_stations_idx = list(main_stations.keys())
# main_stations_idx

# download the zipfile
resp = requests.get(urls[dataset_name])

# we can read the zipfile
stream = io.BytesIO(resp.content)
zf = zipfile.ZipFile(stream)

# this list contains a table of
# station ID, latitude, longitude, station name,
# coastline code, station code, and quality flag
csvtext = zf.read('{}/filelist.txt'.format(dataset_name))

stations = pandas.read_csv(
    io.BytesIO(csvtext),
    sep=';',
    names=('id', 'lat', 'lon', 'name',
           'coastline_code', 'station_code', 'quality'),
    converters={
        'name': str.strip,
        'quality': str.strip
    }
)
stations = stations.set_index('id')

# the dutch stations in the PSMSL database, make a copy
# or use stations.coastline_code == 150 for all dutch stations
selected_stations = stations.ix[main_stations_idx].copy()
# set the main stations, this should be a list of 6 stations
# selected_stations

# each station has a number of files that you can look at.
# here we define a template for each filename

# stations that we are using for our computation
# define the name formats for the relevant files
names = {
    'datum': '{dataset}/RLR_info/{id}.txt',
    'diagram': '{dataset}/RLR_info/{id}.png',
    'url': 'http://www.psmsl.org/data/obtaining/rlr.diagrams/{id}.php',
    'data': '{dataset}/data/{id}.rlrdata',
    'doc': '{dataset}/docu/{id}.txt',
    'contact': '{dataset}/docu/{id}_auth.txt'
}

def get_url(station, dataset):
    """return the url of the station information (diagram and datum)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    url = names['url'].format(**info)
    return url


# fill in the dataset parameter using the global dataset_name
f = functools.partial(get_url, dataset=dataset_name)
# compute the url for each station
selected_stations['url'] = selected_stations.apply(f, axis=1)
# selected_stations


def missing2nan(value, missing=-99999):
    """
    convert the value to nan if the float of value equals the missing value
    """
    value = float(value)
    if value == missing:
        return np.nan
    return value


def year2date(year_fraction, dtype):
    startpoints = np.linspace(0, 1, num=12, endpoint=False)
    remainder = np.mod(year_fraction, 1)
    year = np.floor_divide(year_fraction, 1).astype('int')
    month = np.searchsorted(startpoints, remainder)
    dates = [
        datetime.datetime(year_i, month_i, 1)
        for year_i, month_i in zip(year, month)
    ]
    datetime64s = np.asarray(dates, dtype=dtype)
    return datetime64s


def get_data(station, dataset):
    """get data for the station (pandas record) from the dataset (url)"""
    info = dict(
        dataset=dataset,
        id=station.name
    )
    bytes = zf.read(names['data'].format(**info))
    df = pandas.read_csv(
        io.BytesIO(bytes),
        sep=';',
        names=('year', 'height', 'interpolated', 'flags'),
        converters={
            "height": lambda x: main_stations[station.name]['rlr2nap'](missing2nan(x)),
            "interpolated": str.strip,
        }
    )
    df['station'] = station.name
    df['t'] = year2date(df.year, dtype=wind_df.t.dtype)
    # merge the wind and water levels
    merged = pandas.merge(df, wind_df, how='left', on='t')
    merged['u2'] = np.where(np.isnan(merged['u']),
                            np.nanmean(merged['u']**2),
                            merged['u']**2)
    merged['v2'] = np.where(np.isnan(merged['v']),
                            np.nanmean(merged['v']**2),
                            merged['v']**2)
    return merged

# get data for all stations
f = functools.partial(get_data, dataset=dataset_name)
# look up the data for each station
selected_stations['data'] = [f(station) for _, station in
                             selected_stations.iterrows()]

dfs = []
names = []
for id, station in selected_stations.iterrows():
    df = station['data'].ix[station['data'].year >= 1930]
    dfs.append(df.set_index('year')['height'])
    names.append(station['name'])
merged = pandas.concat(dfs, axis=1)
merged.columns = names
diffs = merged.diff()

df = merged.copy()  # merged.head()

# define the statistical model
def linear_model(df, with_wind=True, with_season=True):
    y = df['height']
    X = np.c_[
        df['year']-1970,
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    month = np.mod(df['year'], 1) * 12.0
    names = ['Constant', 'Trend', 'Nodal U', 'Nodal V']
    if with_wind:
        X = np.c_[
            X,
            df['u2'],
            df['v2']
        ]
        names.extend(['Wind U^2', 'Wind V^2'])
    if with_season:
        for i in range(11):
            X = np.c_[
                X,
                np.logical_and(month >= i, month < i+1)
            ]
            names.append('month_%s' % (i+1, ))
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    fit = model.fit()
    return fit, names

