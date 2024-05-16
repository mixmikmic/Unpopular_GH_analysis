get_ipython().magic('matplotlib inline')
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import urllib
from mpl_toolkits.basemap import Basemap
import ipywidgets as widgets
from ipywidgets import interact
from datetime import datetime, timedelta
from urllib.error import HTTPError

TIMEFRAME = [datetime(2015, 4, 1), datetime(2016, 6, 1)]
SMAP_LOCAL_FILE_URL = "./SMAP/SMAP_L3_SM_P_{}_R13080_001.h5.nc"
SMAP_REMOTE_FILE_URL = "http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/{}/SMAP_L3_SM_P_{}_R13080_001.h5.nc"

#blatantly copied from http://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
def daterange(timeframe):
    for n in range(int ((timeframe[1] - timeframe[0]).days)):
        yield timeframe[0] + timedelta(n)        

def form_smashed_date(date):
    return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
        
def form_dotted_date(date):
    return str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2)

def download_smap_file(date):
    file_name = SMAP_LOCAL_FILE_URL.format(form_smashed_date(date))
    opendap_smap_url = SMAP_REMOTE_FILE_URL.format(form_dotted_date(date), form_smashed_date(date))
    try:
        print("trying to download " + file_name)
        file, headers = urllib.request.urlretrieve(opendap_smap_url, file_name)
    except HTTPError as e:
        print("couldn't download " + file_name + ", " + str(e))

def generate_time_series():
   for date in daterange(TIMEFRAME):
       local_smap_url = SMAP_LOCAL_FILE_URL.format(form_smashed_date(date))
       try:
           dataset = netCDF4.Dataset(local_smap_url)
           get_value(dataset)
       except OSError as e:
           print("oops, couldn't find " + local_smap_url)
           #download_smap_file(date)
            
def get_value(dataset):
   lats = dataset.variables["Soil_Moisture_Retrieval_Data_latitude"][:,:]
   lons = dataset.variables["Soil_Moisture_Retrieval_Data_longitude"][:,:]
    
   simi_valley_lats = (lats >= 34.231) & (lats <= 34.311)
   simi_valley_lons = (lons >= -118.661) & (lons <= -118.869)
    
   row_smla, col_smla = np.where(simi_valley_lats)
   row_smlo, col_smlo = np.where(simi_valley_lons)
    
   mos = dataset.variables["Soil_Moisture_Retrieval_Data_soil_moisture"][row_smla,col_smla]
   write_to_time_series_file(mos)
    
def write_to_time_series_file(writable_data):
   print("Writing...")
   time_series = open("./SMAP_time_series.txt","a")
   time_series.write(str(writable_data[0]))
    
generate_time_series()



