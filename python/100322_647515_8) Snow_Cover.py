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
    
    remote_path = 'ftp://neoftp.sci.gsfc.nasa.gov/geotiff/MOD10C1_M_SNOW/'
    print remote_path

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

    with closing(urllib2.urlopen(remote_path+last_file)) as r:
        with open(last_file, 'wb') as f:
            shutil.copyfileobj(r, f)

    with rasterio.open(local_path+'/'+last_file) as src:
        npixels = src.width * src.height
        for i in src.indexes:
            band = src.read(i)
            print(i, band.min(), band.max(), band.sum()/npixels)

    
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
outFile = 'snow_cover.tiff'

print 'starting'
file = dataDownload()
print 'downloaded'
tiffile(file,outFile)
print 'converted'
#s3Upload(outFile)
print 'finish'

