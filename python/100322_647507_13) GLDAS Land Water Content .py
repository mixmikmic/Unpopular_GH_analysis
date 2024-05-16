import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import urllib2
from contextlib import closing
import rasterio
import os
import shutil
import netCDF4
import scipy
from scipy import ndimage
get_ipython().magic('matplotlib inline')

remote_path = 'ftp://podaac-ftp.jpl.nasa.gov/allData/tellus/L3/land_mass/RL05/netcdf/'
local_path = os.getcwd()

listing = []
response = urllib2.urlopen(remote_path)
for line in response:
    listing.append(line.rstrip())

s2=pd.DataFrame(listing)
s3=s2[0].str.split()
s4=s3[len(s3)-1]
last_file = s4[8]
print 'The last file is: ',last_file

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
ncfile.variables['lwe_thickness'][:]

# open a local NetCDF file or remote OPeNDAP URL
url = local_path+'/'+last_file
nc = netCDF4.Dataset(url)

# examine the variables
print nc.variables.keys()
print nc.variables['lwe_thickness']

# Data from variable of interest
topo = nc.variables['lwe_thickness'][1,:,:]


# Ploting
plt.figure(figsize=(10,10))
plt.imshow(topo)

rows, columns = topo.shape              # get sizes

# Reverse the array
flipped_array = np.fliplr(topo) 

left_side = topo[:,int(columns/2):]     # split the array... 
right_side = topo[:,:int(columns/2)]    # ...into two halves. Then recombine.
wsg84_array = np.concatenate((left_side,right_side), axis=1)

#reverse again
a = scipy.ndimage.interpolation.rotate(wsg84_array, 180)
fliped = np.fliplr(a)
plt.figure(figsize=(10,10))
plt.imshow(fliped, cmap=cm.jet)

import numpy as np
import pandas as pd
import os
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import rasterio
import tinys3
import scipy
from scipy import ndimage
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm
np.set_printoptions(threshold='nan')

def dataDownload(): 
    
    remote_path = 'ftp://podaac-ftp.jpl.nasa.gov/allData/tellus/L3/land_mass/RL05/netcdf/'
    local_path = os.getcwd()

    listing = []
    response = urllib2.urlopen(remote_path)
    for line in response:
        listing.append(line.rstrip())

    s2=pd.DataFrame(listing)
    s3=s2[0].str.split()
    s4=s3[len(s3)-1]
    last_file = s4[8]
    print 'The last file is: ',last_file

    print (remote_path)
    print (last_file)
    print (local_path)

    #Download the file .nc
    with closing(urllib2.urlopen(remote_path+'/'+last_file)) as r:
        with open(str(last_file), 'wb') as f:
            shutil.copyfileobj(r, f)

    ncfile = Dataset(local_path+'/'+last_file)
    
    return last_file

def netcdf2tif(dst,outFile):
    nc = Dataset(dst)
    data = nc['lwe_thickness'][1,:,:]
            
    data[data < 0] = -1
    data[data == 32767.0] = -1
    
    print data
    
    # Return lat info
    south_lat = -90
    north_lat = 90

    # Return lon info
    west_lon = -180
    east_lon = 180
    
    rows, columns = data.shape              # get sizes

    # Reverse the array
    flipped_array = np.fliplr(data) 

    left_side = data[:,int(columns/2):]     # split the array... 
    right_side = data[:,:int(columns/2)]    # ...into two halves. Then recombine.
    wsg84_array = np.concatenate((left_side,right_side), axis=1)

    #reverse again
    a = scipy.ndimage.interpolation.rotate(wsg84_array, 180)
    fliped = np.fliplr(a)
    #plt.figure(figsize=(10,10))
    #plt.imshow(fliped, cmap=cm.jet)  
    
    print 'transformation.......'
    # Transformation function
    transform = rasterio.transform.from_bounds(west_lon, south_lat, east_lon, north_lat, columns, rows)
    # Profile
    profile = {
        'driver':'GTiff', 
        'height':rows, 
        'width':columns, 
        'count':1, 
        'dtype':np.float64, 
        'crs':'EPSG:4326', 
        'transform':transform, 
        'compress':'lzw', 
        'nodata': -1
    }
    with rasterio.open(outFile, 'w', **profile) as dst:
        dst.write(fliped.astype(profile['dtype']), 1)
    
    print 'Data Shape: ',columns
    print 'Data Shape: ',rows
    os.remove('./'+file)

def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))

# Execution
outFile ='land_water.tif'
print 'starting'
file = dataDownload()
print 'downloaded'
netcdf2tif(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'

scipy.__version__

