import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import urllib2
from contextlib import closing
import rasterio
import os
import shutil
import netCDF4
get_ipython().magic('matplotlib inline')
np.set_printoptions(threshold='nan')

remote_path = 'ftp://aftp.cmdl.noaa.gov/products/carbontracker/ch4/fluxes/'
last_file = '201012.nc'

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

ncfile.info()

ncfile.variables

# open a local NetCDF file or remote OPeNDAP URL
url = url=local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['fossil']

# sample every 10th point of the 'z' variable
fossil = nc.variables['fossil'][0,:,:]
print 'Shape: ',fossil.shape
agwaste = nc.variables['agwaste'][0,:,:]
print 'Shape: ',agwaste.shape
natural = nc.variables['natural'][0,:,:]
print 'Shape: ',natural.shape
bioburn = nc.variables['bioburn'][0,:,:]
print 'Shape: ',bioburn.shape
ocean = nc.variables['ocean'][0,:,:]
print 'Shape: ',ocean.shape

topo = [a + b + c + d + e for a, b, c, d, e in zip(fossil, agwaste, natural, bioburn, ocean)]
#print topo

# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo,clim=(0.0, 1))

for i in reversed(topo):
    data = list(reversed(topo))

plt.figure(figsize=(10,10))
plt.imshow(data,clim=(0.0, 200))

import numpy as np
from contextlib import closing
import urllib2
import shutil
import os
from netCDF4 import Dataset
import rasterio
import tinys3
import netCDF4

def dataDownload(): 
    remote_path = 'ftp://aftp.cmdl.noaa.gov/products/carbontracker/ch4/fluxes/'
    last_file = '201012.nc'

    local_path = os.getcwd()

    print (remote_path)
    print (last_file)
    print (local_path)

    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(last_file, 'wb') as f:
            shutil.copyfileobj(r, f)
    
    return last_file

def netcdf2tif(dst,outFile):
    local_path = os.getcwd()
    url = local_path+'/'+dst
    nc = netCDF4.Dataset(url)

    # examine the variables
    print nc.variables.keys()

    # sample every 10th point of the 'z' variable
    fossil = nc.variables['fossil'][0,:,:]
    print 'Shape: ',fossil.shape
    agwaste = nc.variables['agwaste'][0,:,:]
    print 'Shape: ',agwaste.shape
    natural = nc.variables['natural'][0,:,:]
    print 'Shape: ',natural.shape
    bioburn = nc.variables['bioburn'][0,:,:]
    print 'Shape: ',bioburn.shape
    ocean = nc.variables['ocean'][0,:,:]
    print 'Shape: ',ocean.shape

    topo = [a + b + c + d + e for a, b, c, d, e in zip(fossil, agwaste, natural, bioburn, ocean)]

    for i in reversed(topo):
        data = list(reversed(topo))
    
    data = np.asarray(data)
    
    data[data < 0] = -1

    # Return lat info
    south_lat = -90
    north_lat = 90

    # Return lon info
    west_lon = -180
    east_lon = 180
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

def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))

# Execution
outFile ='methane.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'



