import numpy as np
import os
import urllib2
import shutil
from contextlib import closing
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import netCDF4
import matplotlib
np.set_printoptions(threshold='nan')

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/kaplan_sst/'
last_file = 'sst.mon.anom.nc'
local_path = os.getcwd()

#Download the file .nc
with closing(urllib2.urlopen(remote_path+'/'+last_file)) as r:
    with open(str(last_file), 'wb') as f:
        shutil.copyfileobj(r, f)

ncfile = xr.open_dataset(local_path+'/'+last_file, decode_times=False)

print('* Variables disponibles en el fichero:')
for v in ncfile.variables:
    print(v)

print ncfile.variables['sst']

ncfile.info()

local_path = os.getcwd()


# set up the figure
plt.figure(figsize=(16,12))

url=local_path+'/'+last_file

# Extract the significant 

file = netCDF4.Dataset(url)
lat  = file.variables['lat'][:]
lon  = file.variables['lon'][:]
data = file.variables['sst'][1,:,:]
file.close()

m=Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0,
              lat_0=-87.5, lon_0=2.5)

# convert the lat/lon values to x/y projections.

x, y = m(*np.meshgrid(lon,lat))
m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.jet)
m.colorbar(location='right')

# Add a coastline and axis values.

m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-87.5,87.5,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(2.5,357.5,60.),labels=[0,0,0,1])

plt.show()

# make image
plt.imshow(data,origin='lower') 

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/kaplan_sst/'
last_file = 'sst.mon.anom.nc'
local_path = os.getcwd()

# open a local NetCDF file or remote OPeNDAP URL
url = local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['sst']

topo = nc.variables['sst'][0,:,:]

# make image
plt.figure(figsize=(10,10))
plt.imshow(topo)
plt.show()

import numpy as np
import os
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import rasterio
import tinys3
np.set_printoptions(threshold='nan')

def dataDownload(): 
    remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/kaplan_sst/'
    last_file = 'sst.mon.anom.nc'
    local_path = os.getcwd()
    print remote_path
    print last_file
    print local_path

    #Download the file .nc
    with closing(urllib2.urlopen(remote_path+'/'+last_file)) as r:
        with open(str(last_file), 'wb') as f:
            shutil.copyfileobj(r, f)

    ncfile = Dataset(local_path+'/'+last_file)
    
    return last_file

def netcdf2tif(dst,outFile):
    nc = Dataset(dst)
    data = nc['sst'][0,:,:]
            
    data[data < -8] = -99
    data[data > 8] = -99
    # Converting in to zero for the output raster
    #np.putmask(data, data < -8, -99)
    
    print data
    
    # Return lat info
    south_lat = -88.75
    north_lat = 88.75

    # Return lon info
    west_lon = -177.5
    east_lon = 177.5
    # Transformation function
    transform = rasterio.transform.from_bounds(west_lon, south_lat, east_lon, north_lat, data.shape[1], data.shape[0])
    # Profile
    profile = {
        'driver':'GTiff', 
        'height':data.shape[0], 
        'width':data.shape[1], 
        'count':1, 
        'dtype':np.float64, 
        'crs':'EPSG:4326', 
        'transform':transform, 
        'compress':'lzw', 
        'nodata': -99
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)

def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))

# Execution
outFile ='ssta.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'

src = rasterio.open('./'+outFile)
print 'Source: ',src
print 'Source mode: ',src.mode

array = src.read(1)
print '.TIF Shape: ',array.shape

print 'Source type:',src.dtypes
print(src.crs)
print(src.transform)

from matplotlib import pyplot
pyplot.imshow(array, cmap='gist_earth')

pyplot.show()



