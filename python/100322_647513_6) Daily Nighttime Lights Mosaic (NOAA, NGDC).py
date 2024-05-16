import pandas as pd
import numpy as np
from os.path import basename, dirname, exists
import os
import rasterio
import glob
import urllib2
import gzip
import shutil
from contextlib import closing
from matplotlib import pyplot
#from netCDF4 import Dataset

remote_path = 'https://www.nasa.gov/specials/blackmarble/2016/globalmaps/georeferrenced/'
last_file = 'BlackMarble_2016_3km_geo.tif'


local_path = os.getcwd()
print remote_path+last_file

with closing(urllib2.urlopen(remote_path_land_ocean+land_ocean_file)) as r:
    with open(local_path+'/'+last_file, 'wb') as f:
        shutil.copyfileobj(r, f)  

src = rasterio.open(local_path+'/'+last_file)

print 'Source: ',src
print 'Source mode: ',src.mode

array = src.read(1)
print '.TIF Shape: ',array.shape

print 'Source type:',src.dtypes
print(src.crs)
print(src.transform)

pyplot.imshow(array, cmap='RdYlBu_r')
pyplot.show()

with rasterio.open(local_path+'/'+last_file) as src:
    npixels = src.width * src.height
    for i in src.indexes:
        band = src.read(i)
        print(i, band.min(), band.max(), band.sum()/npixels)

CM_IN_FOOT = 30.48

with rasterio.drivers():
    with rasterio.open(local_path+'/'+last_file) as src:
        kwargs = src.meta
        kwargs.update(
            driver='GTiff',
            dtype=rasterio.float64,  #rasterio.int16, rasterio.int32, rasterio.uint8,rasterio.uint16, rasterio.uint32, rasterio.float32, rasterio.float64
            count=1,
            compress='lzw',
            nodata=0,
            bigtiff='NO' # Output will be larger than 4GB
        )

        windows = src.block_windows(1)

        with rasterio.open(local_path+'/'+last_file,'w',**kwargs) as dst:
            for idx, window in windows:
                src_data = src.read(1, window=window)

                # Source nodata value is a very small negative number
                # Converting in to zero for the output raster
                np.putmask(src_data, src_data < 0, 0)

                dst_data = (src_data * CM_IN_FOOT).astype(rasterio.float64)
                dst.write_band(1, dst_data, window=window)

src = rasterio.open(local_path+'/'+last_file)
print 'Source: ',src
print 'Source mode: ',src.mode

array = src.read(1)
print '.TIF Shape: ',array.shape

print 'Source type:',src.dtypes
print(src.crs)
print(src.transform)

from matplotlib import pyplot
pyplot.imshow(array, cmap='RdYlBu_r')

pyplot.show()

import numpy as np
import pandas as pd
import os
import rasterio
import urllib2
import shutil
from contextlib import closing
from netCDF4 import Dataset
import datetime
import tinys3
np.set_printoptions(threshold='nan')

def dataDownload(): 
    
    remote_path = 'https://www.nasa.gov/specials/blackmarble/2016/globalmaps/georeferrenced/BlackMarble_2016_3km_geo.tif'
    last_file = 'BlackMarble_2016_3km_geo.tif'

    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(str(last_file), 'wb') as f:
            shutil.copyfileobj(r, f)

    return last_file

def tiffile(dst,outFile):
    
    CM_IN_FOOT = 30.48


    with rasterio.open(file) as src:
        kwargs = src.meta
        kwargs.update(
            driver='GTiff',
            dtype=rasterio.float64,  #rasterio.int16, rasterio.int32, rasterio.uint8,rasterio.uint16, rasterio.uint32, rasterio.float32, rasterio.float64
            count=1,
            compress='lzw',
            nodata=0,
            bigtiff='NO' 
        )

        windows = src.block_windows(1)

        with rasterio.open(outFile,'w',**kwargs) as dst:
            for idx, window in windows:
                src_data = src.read(1, window=window)

                # Source nodata value is a very small negative number
                # Converting in to zero for the output raster
                np.putmask(src_data, src_data < 0, 0)

                dst_data = (src_data * CM_IN_FOOT).astype(rasterio.float64)
                dst.write_band(1, dst_data, window=window)
    os.remove('./'+file)

def s3Upload(outFile):
    # Push to Amazon S3 instance
    conn = tinys3.Connection(os.getenv('S3_ACCESS_KEY'),os.getenv('S3_SECRET_KEY'),tls=True)
    f = open(outFile,'rb')
    conn.upload(outFile,f,os.getenv('BUCKET'))

# Execution
outFile = 'earth_ligths.tif'

print 'starting'
file = dataDownload()
print 'downloaded'
tiffile(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'





