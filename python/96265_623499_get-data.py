# this is a list of packages that are used in this notebook
# these come with python
import io
import zipfile
import functools
import bisect
import datetime
import re

# you can install these packages using pip or anaconda
# (requests numpy pandas bokeh pyproj statsmodels)

# for downloading
import requests
import netCDF4

# computation libraries
import numpy as np
import pandas

# statistics
import statsmodels.api as sm

# Define the urls for the three PSMSL datasets
urls = {
    'met_monthly': 'http://www.psmsl.org/data/obtaining/met.monthly.data/met_monthly.zip',
    'rlr_monthly': 'http://www.psmsl.org/data/obtaining/rlr.monthly.data/rlr_monthly.zip',
    'rlr_annual': 'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip'
}

# each station has a number of files that you can look at.
# here we define a template for each filename
names = {
    'datum': '{dataset}/RLR_info/{id}.txt',
    'diagram': '{dataset}/RLR_info/{id}.png',
    'url': 'http://www.psmsl.org/data/obtaining/rlr.diagrams/{id}.php',
    'data': '{dataset}/data/{id}.{typetag}data',
    'doc': '{dataset}/docu/{id}.txt',
    'contact': '{dataset}/docu/{id}_auth.txt',
    'rlr_info': '{dataset}/RLR_info/{id}.txt',
}

def make_wind_df(lat_i=53, lon_i=3):
    """
    Create a dataset for wind, for 1 latitude/longitude
    
    Parameters
    ----------
    lat_i : int
        degree latitude
    lon_i : int
        degree longitude
    """
    u_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/uwnd.10m.mon.mean.nc'
    v_file = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/vwnd.10m.mon.mean.nc'

    # open the 2 files
    ds_u = netCDF4.Dataset(u_file)
    ds_v = netCDF4.Dataset(v_file)
    # read lat,lon, time from 1 dataset
    lat, lon, time = ds_u.variables['lat'][:], ds_u.variables['lon'][:], ds_u.variables['time'][:]
    # check with the others
    lat_v, lon_v, time_v = ds_v.variables['lat'][:], ds_v.variables['lon'][:], ds_v.variables['time'][:]
    assert (lat == lat_v).all() and (lon == lon_v).all() and (time == time_v).all()
    # convert to datetime
    t = netCDF4.num2date(time, ds_u.variables['time'].units)
    
    def find_closest(lat, lon, lat_i=lat_i, lon_i=lon_i):
        """lookup the index of the closest lat/lon"""
        Lon, Lat = np.meshgrid(lon, lat)
        idx = np.argmin(((Lat - lat_i)**2 + (Lon - lon_i)**2))
        Lat.ravel()[idx], Lon.ravel()[idx]
        [i, j] = np.unravel_index(idx, Lat.shape)
        return i, j
    # this is the index where we want our data
    i, j = find_closest(lat, lon)
    # get the u, v variables
    print('found point', lat[i], lon[j])
    u = ds_u.variables['uwnd'][:, i, j]
    v = ds_v.variables['vwnd'][:, i, j]
    # compute derived quantities
    speed = np.sqrt(u ** 2 + v **2)
    # compute direction in 0-2pi domain
    direction = np.mod(np.angle(u + v * 1j), 2*np.pi)
    # put everything in a dataframe
    wind_df = pandas.DataFrame(data=dict(u=u, v=v, t=t, speed=speed, direction=direction))
    # return it
    return wind_df

def get_stations(zf, dataset_name, coastline_code=150):
    """
    Function to get a dataframe with the tide gauge stations within a dataset.
    The stations are filtered on a certain coastline_code, indicating a country.
    
    Parameters
    ----------
    zf : zipfile.ZipFile
        Downloaded zipfile
    dataset_name : string
        Name of the dataset that is used: met_monthly, rlr_monthly, rlr_annual
    coastline_code : int
        Coastline code indicating the country
    """
    # this list contains a table of 
    # station ID, latitude, longitude, station name, coastline code, station code, and quality flag
    csvtext = zf.read('{}/filelist.txt'.format(dataset_name))
    
    # Read the stations from the comma seperated text.
    stations = pandas.read_csv(
        io.BytesIO(csvtext), 
        sep=';',
        names=('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality'),
        converters={
            'name': str.strip,
            'quality': str.strip
        }
    )
    # Set index on column 'id'
    stations = stations.set_index('id')
    
    # filter on coastline code (Netherlands is 150)
    selected_stations = stations.where(stations['coastline_code'] == coastline_code).dropna(how='all')
    
    return selected_stations

def get_url(station, dataset):
    """return the url of the station information (diagram and datum)"""
    print(dataset, station.name, dataset.split('_')[0])
    info = dict(
        dataset=dataset,
        id=station.name,
        typetag=dataset.split('_')[0]
    )
    url = names['url'].format(**info)
    return url

def missing2nan(value, missing=-99999):
    """convert the value to nan if the float of value equals the missing value"""
    value = float(value)
    if value == missing:
        return np.nan
    return value

def year2date(year_fraction, dtype):
    """convert a year fraction to a datetime"""
    startpoints = np.linspace(0, 1, num=12, endpoint=False)
    remainder = np.mod(year_fraction, 1)
    year = np.floor_divide(year_fraction, 1).astype('int')
    month = np.searchsorted(startpoints, remainder)
    dates = [
        datetime.datetime(year_i, month_i, 1) 
        for year_i, month_i 
        in zip(year, month)
    ]
    datetime64s = np.asarray(dates, dtype=dtype)
    return datetime64s

def get_rlr2nap(zf, station, dataset):
    """
    Read rlr 2 nap correction from zipfile
    """
    info = dict(
        dataset=dataset,
        id=station.name,
    )
    
    bytes = zf.read(names['rlr_info'].format(**info))
    correction = float(re.findall('Add (.+) to data .+ onwards', bytes.decode())[0].replace('m', '')) * 1000
    
    return lambda x: x - correction
    

def get_data(zf, wind_df, station, dataset):
    """
    get data for the station (pandas record) from the dataset (url)
    
    Parameters
    ----------
    zf : zipfile.ZipFile
        Downloaded zipfile to get the data from
    wind_df : pandas.DataFrame
        Dataset with the wind for a given latitude and longitude
    station : pandas.Series
        Row of the selected_stations dataframe with station meta data
    dataset : string
        Name of the data set    
    """
    # rlr or met
    typetag=dataset.split('_')[0]
    
    info = dict(
        dataset=dataset,
        id=station.name,
        typetag=typetag
    )
    bytes = zf.read(names['data'].format(**info))
    converters = {
            "interpolated": str.strip,
        }
    if typetag == 'rlr':
        rlr2nap = get_rlr2nap(zf, station, dataset)
        converters['height'] = lambda x: rlr2nap(missing2nan(x))
        
    df = pandas.read_csv(
        io.BytesIO(bytes), 
        sep=';', 
        names=('year', 'height', 'interpolated', 'flags'),
        converters=converters,
    )
    df['station'] = station.name
    df['t'] = year2date(df.year, dtype=wind_df.t.dtype)
    # merge the wind and water levels
    merged = pandas.merge(df, wind_df, how='left', on='t')
    merged['u2'] = np.where(np.isnan(merged['u']), np.nanmean(merged['u']**2), merged['u']**2)
    merged['v2'] = np.where(np.isnan(merged['v']), np.nanmean(merged['v']**2), merged['v']**2)
    return merged

def get_station_data(dataset_name, coastline_code=150):
    """Method to get the station data for a certain dataset"""

    # download the zipfile
    resp = requests.get(urls[dataset_name])

    wind_df = make_wind_df()
      
    # we can read the zipfile
    stream = io.BytesIO(resp.content)
    zf = zipfile.ZipFile(stream)
    
    selected_stations = get_stations(zf, dataset_name=dataset_name, coastline_code=coastline_code)
    # fill in the dataset parameter using the global dataset_name
    f = functools.partial(get_url, dataset=dataset_name)
    # compute the url for each station
    selected_stations['url'] = selected_stations.apply(f, axis=1)
    
    selected_stations['data'] = [get_data(zf, wind_df, station, dataset=dataset_name) for _, station in selected_stations.iterrows()]
   
    return selected_stations

def linear_model(df, with_wind=True, with_season=True):
    """
    Return the fit from the linear model on the given dataset df.
    Wind and season can be enabled and disabled
    """
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



