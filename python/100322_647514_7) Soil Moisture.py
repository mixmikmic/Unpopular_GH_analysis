import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import urllib2
from contextlib import closing
from matplotlib.pyplot import cm
import rasterio
import os
import shutil
import netCDF4
get_ipython().magic('matplotlib inline')

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cpcsoil/'
last_file = 'soilw.mon.ltm.v2.nc'

local_path = os.getcwd()

print (remote_path)
print (last_file)
print (local_path)

with closing(urllib2.urlopen(remote_path+last_file)) as r:
    with open(last_file, 'wb') as f:
        shutil.copyfileobj(r, f)

ncfile = xr.open_dataset(local_path+'/'+last_file, decode_times=False)

print('* Variables disponibles en el fichero:')
for v in ncfile.variables:
    print(v)

#Con este comando vemos la info general del fichero .nc 
ncfile.info()

#info de la variable precip
ncfile.variables['soilw'][:]

# open a local NetCDF file or remote OPeNDAP URL
url = url=local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['soilw']

# Data from variable of interest
topo = nc.variables['soilw'][1,:,:]

# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)

rows, columns = topo.shape              # get sizes
print rows
print columns

flipped_array = np.fliplr(topo)   # Reverse the array

left_side = topo[:,int(columns/2):]     # split the array... 
right_side = topo[:,:int(columns/2)]    # ...into two halves. Then recombine.
wsg84_array = np.concatenate((topo[:,int(columns/2):],topo[:,:int(columns/2)]), axis=1)
print(wsg84_array.shape)                         #  confirm we havent screwed the size of the array
plt.figure(figsize=(10,10))
plt.imshow(wsg84_array, cmap=cm.jet, vmin=1.86264515e-06, vmax=7.43505005e+02)

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
    
    remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cpcsoil/'
    last_file = 'soilw.mon.ltm.v2.nc'
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
    data = nc['soilw'][1,:,:]
            
    data[data < 0] = -1
    data[data > 1000] = -1
    
    print data
    
    # Return lat info
    south_lat = -89.75
    north_lat = 89.75

    # Return lon info
    west_lon = 0.25
    east_lon = 359.75
    
    
    rows, columns = data.shape              # get sizes
    print rows
    print columns
    flipped_array = np.fliplr(data)
    left_side = data[:,int(columns/2):]     # split the array... 
    right_side = data[:,:int(columns/2)]    # ...into two halves. Then recombine.
    wsg84_array = np.concatenate((data[:,int(columns/2):],data[:,:int(columns/2)]), axis=1)
    
    
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
        'nodata': -1
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)
    
    print 'Data Shape: ',data.shape[1]
    print 'Data Shape: ',data.shape[0]

def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))

# Execution
outFile ='soil_moisture.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'

