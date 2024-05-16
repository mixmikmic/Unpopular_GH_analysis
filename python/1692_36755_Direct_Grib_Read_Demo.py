from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import glob
import mmmpy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import StamenTerrain
import pygrib
import os
import pyart
get_ipython().magic('matplotlib inline')

def download_files(input_dt, max_seconds=300):
    """
    This function takes an input datetime object, and will try to match with the closest mosaics in time
    that are available at NCEP. Note that NCEP does not archive much beyond 24 hours of data.
    
    Parameters
    ----------
    input_dt : datetime.datetime object
        input datetime object, will try to find closest file in time on NCEP server
    
    Other Parameters
    ----------------
    max_seconds : int or float
        Maximum number of seconds difference tolerated between input and selected datetimes,
        before file matching will fail
    
    Returns
    -------
    files : 1-D ndarray of strings
        Array of mosaic file names, ready for ingest into MMM-Py
    """
    baseurl = 'http://mrms.ncep.noaa.gov/data/3DReflPlus/'
    page1 = pd.read_html(baseurl)
    directories = np.array(page1[0][0][2:])
    urllist = []
    files = []
    for i, d in enumerate(directories):
        page2 = pd.read_html(baseurl + d)
        filelist = np.array(page2[0][0][2:])
        dts = []
        for filen in filelist:
            # Will need to change in event of a name change
            dts.append(dt.datetime.strptime(filen[32:47], '%Y%m%d-%H%M%S'))
        dts = np.array(dts)
        diff = np.abs((dts - input_dt))
        if np.min(diff).total_seconds() <= max_seconds:
            urllist.append(baseurl + d + filelist[np.argmin(diff)])
            files.append(filelist[np.argmin(diff)])
    for url in urllist:
        print(url)
        os.system('wget ' + url)
    return np.array(files)

files = download_files(dt.datetime.utcnow())

files

mosaic = mmmpy.MosaicTile(files)

mosaic.diag()

tiler = StamenTerrain()

ext = [-130, -65, 20, 50]
fig = plt.figure(figsize=(12, 6))
projection = ccrs.PlateCarree()  # ShadedReliefESRI().crs
ax = plt.axes(projection=projection)
ax.set_extent(ext)
ax.add_image(tiler, 3)

# Create a feature for States/Admin 1 regions at 1:10m from Natural Earth
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
ax.add_feature(states_provinces, edgecolor='gray')

# Create a feature for Countries 0 regions at 1:10m from Natural Earth
countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='50m',
    facecolor='none')
ax.add_feature(countries, edgecolor='k')

ax.coastlines(resolution='50m')

mosaic.get_comp()
valmask = np.ma.masked_where(mosaic.mrefl3d_comp <= 0, mosaic.mrefl3d_comp)
cs = plt.pcolormesh(mosaic.Longitude, mosaic.Latitude, valmask, vmin=0, vmax=55,
                    cmap='pyart_Carbone42', transform=projection)
plt.colorbar(cs, label='Composite Reflectivity (dBZ)',
             orientation='horizontal', pad=0.05, shrink=0.75, fraction=0.05, aspect=30)
plt.title(dt.datetime.utcfromtimestamp(mosaic.Time).strftime('%m/%d/%Y %H:%M UTC'))







