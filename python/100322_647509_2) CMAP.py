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
import matplotlib.pyplot as plt
import netCDF4
get_ipython().magic('matplotlib inline')

remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cmap/enh/'
last_file = 'precip.mon.ltm.nc'

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

#General Info of .nc file
ncfile.info()

# Variables ifno
ncfile.variables

#info de la variable precip
ncfile.variables['precip'][:]

# open a local NetCDF file or remote OPeNDAP URL
url =local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['precip']

# sample every 10th point of the 'z' variable
topo = nc.variables['precip'][0,:,:]
print topo

idxs = np.where([val == -9.96921e+36 for val in topo])[0]
topo[idxs] = -1

# make image
plt.figure(figsize=(10,10))
plt.imshow(topo)

import numpy as np
from contextlib import closing
import urllib2
import shutil
import os
from netCDF4 import Dataset
import rasterio
import tinys3

def dataDownload(): 
    remote_path = 'ftp://ftp.cdc.noaa.gov/Datasets/cmap/enh/'
    last_file = 'precip.mon.ltm.nc'

    local_path = os.getcwd()

    print (remote_path)
    print (last_file)
    print (local_path)

    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(last_file, 'wb') as f:
            shutil.copyfileobj(r, f)

    ncfile = Dataset(local_path+'/'+last_file)
    
    return last_file

def netcdf2tif(dst,outFile):
    nc = Dataset(dst)
    data = nc['precip'][0,:,:]
    
    #data[data == -9.96921e+36] = -1
    idxs = np.where([val == -9.96921e+36 for val in data])[0]
    data[idxs] = -1

    print data
    
    # Return lat info
    south_lat = -88.75
    north_lat = 88.75

    # Return lon info
    west_lon = -178.75
    east_lon = 178.75
    # Transformation function
    transform = rasterio.transform.from_bounds(west_lon, south_lat, east_lon, north_lat, data.shape[1], data.shape[0])
    # Profile
    profile = {
        'driver':'GTiff', 
        'height':data.shape[0], 
        'width':data.shape[1], 
        'count':1, 
        'dtype':np.int16, 
        'crs':'EPSG:4326', 
        'transform':transform, 
        'compress':'lzw', 
        'nodata': -1
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(data.astype(profile['dtype']), 1)

def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))

# Execution
outFile ='cmap.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'



