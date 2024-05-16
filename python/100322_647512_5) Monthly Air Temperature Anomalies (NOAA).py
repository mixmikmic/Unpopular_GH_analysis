import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import urllib2
from contextlib import closing
from matplotlib.pyplot import cm
import matplotlib.image as mpimg
import rasterio
import os
import shutil
import netCDF4
get_ipython().magic('matplotlib inline')

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/noaaglobaltemp/'
last_file = 'air.mon.anom.nc'
local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)

with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)

ncfile = xr.open_dataset(local_path+'/'+last_file, decode_times=False)

#To see availables variables in the file
print('* Variables disponibles en el fichero:')
for v in ncfile.variables:
    print(v)

#To see general info of the .nc file 
ncfile.info()

# open a local NetCDF file
url = local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine once again to be sure the variables
print nc.variables.keys()
print nc.variables['air']

# Taking the data
topo = nc.variables['air'][1,:,:]

# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)

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
    remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/noaaglobaltemp/'
    last_file = 'air.mon.anom.nc'

    local_path = os.getcwd()

    print (remote_path)
    print (last_file)
    print (local_path)

    #Download the file .nc
    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(last_file, 'wb') as f:
            shutil.copyfileobj(r, f)

    ncfile = Dataset(local_path+'/'+last_file)
    
    return last_file

def netcdf2tif(dst,outFile):
    nc = Dataset(dst)
    data = nc['air'][1,:,:]
            
    data[data < -40] = -99
    data[data > 40] = -99
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
        'nodata':-99
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)

def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))

# Execution
outFile ='air_temo_anomalies.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cru/crutem4/std/'
last_file = 'air.mon.anom.nc'

local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)

with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)
        
# open a local NetCDF file
url = url=local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['air']

# Selecting data
topo = nc.variables['air'][1,:,:]

# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cru/hadcrut4/'
last_file = 'air.mon.anom.median.nc'

local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)

with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)
        
# open a local NetCDF file
url = url=local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['air']

# Data
topo = nc.variables['air'][1,:,:]

# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)

