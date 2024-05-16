import netCDF4
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import time
from urllib.error import HTTPError
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap

#soil moisture dataset
SMAP_DATASET_URL = "http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/"
SMAP_FILE_URL = "/SMAP_L3_SM_P_{}_R13080_001.h5"
SMAP_VARIABLES = "?Soil_Moisture_Retrieval_Data_soil_moisture[0:1:405][0:1:963],Soil_Moisture_Retrieval_Data_latitude[0:1:405][0:1:963],Soil_Moisture_Retrieval_Data_longitude[0:1:405][0:1:963]"

#vegetation greenery dataset
MODIS_DATASET_URL = "http://opendap.cr.usgs.gov:80/opendap/hyrax/MOD13A2.006/"
MODIS_VARIABLES = "?Latitude[0:1:1199][0:1:1199],Longitude[0:1:1199][0:1:1199],_1_km_16_days_NDVI[0:1:376][0:1:1199][0:1:1199]"
#vegetation classification dataset
MCD_DATASET_URL = "http://opendap.cr.usgs.gov:80/opendap/hyrax/MCD12Q1.051/"
MCD_VARIABLES = "?Latitude[0:1:2399][0:1:2399],Longitude[0:1:2399][0:1:2399],Land_Cover_Type_2[0:1:12][0:1:2399][0:1:2399]"
#same filenames apply to both
MODIS_MCD_FILE_URLS = ["h08v04.ncml", "h08v05.ncml", "h09v04.ncml", "h09v05.ncml", "h09v06.ncml", "h10v04.ncml", "h10v05.ncml", 
                   "h10v06.ncml", "h11v04.ncml", "h11v05.ncml", "h12v04.ncml", "h12v05.ncml", "h13v04.ncml"]

TIMEFRAME = [datetime(2015, 4, 1), datetime(2016, 6, 1)]

#blatantly copied from http://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
def daterange(timeframe):
    for n in range(int ((timeframe[1] - timeframe[0]).days)):
        yield timeframe[0] + timedelta(n)        

def form_smashed_date(date):
    return str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
        
def form_dotted_date(date):
    return str(date.year) + "." + str(date.month).zfill(2) + "." + str(date.day).zfill(2)

def get_file(file_url, file_name, dataset_type):
    try:
        file, headers = urllib.request.urlretrieve(file_url, dataset_type + "/" + file_name)
    except HTTPError as e:
        print("There was an error: " + str(e.code))
        return None
    else:
        return file

def ingest_smap(date):
    formed_url = SMAP_DATASET_URL + form_dotted_date(date) + SMAP_FILE_URL.format(form_smashed_date(date))
    print(formed_url)
    dataset = netCDF4.Dataset(formed_url)
    print("loaded dataset " + SMAP_FILE_URL.format(form_smashed_date(date)))
    return dataset
    
def ingest_modis(name):
    dataset = netCDF4.Dataset(MODIS_DATASET_URL + name) 
    print("Loaded dataset " + name)
    return dataset
    
def ingest_mcd(name):
    dataset = netCDF4.Dataset(MCD_DATASET_URL + name)
    print("Loaded dataset " + name)
    return dataset
    
def average_smap():
    print("doo some stuff")
    
def graph_data():
    print("doo some stuff")

get_ipython().magic('matplotlib inline')
def proccess_smap_dataset(basemap, dataset):
    print("processing smap dataset")
    start = time.clock()
    lat = dataset.variables["Soil_Moisture_Retrieval_Data_latitude"][:, :]
    lon = dataset.variables["Soil_Moisture_Retrieval_Data_longitude"][:, :]
    mos = dataset.variables["Soil_Moisture_Retrieval_Data_soil_moisture"][:, :]
    end = time.clock()
    print("drawing smap dataset, processing took " + str(end-start))
    #basemap.pcolormesh(lon, lat, mos, latlon=True)
    basemap.pcolor(lon, lat, mos, latlon=True)
    
def proccess_modis_mcd_dataset(basemap, dataset, name):
    print("processing dataset " + name)
    start = time.clock()
    basemap.pcolor(dataset.variables["Latitude"][:, :], dataset.variables["Longitude"][:, :], dataset.variables["_1_km_16_days_NDVI"][0, :, :], latlon=True)
    end = time.clock()
    print("drawing dataset " + name + ", processing took " + str(end-start))
    
def main():
    plot = plt.figure(figsize=(15,15))
    m = Basemap(projection='ortho',lat_0=20,lon_0=-100,resolution='c')
    #m = Basemap(projection='robin',lon_0=0,resolution='c')
    m.drawcoastlines()
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    for date in daterange(TIMEFRAME):
        d = ingest_smap(date)
        proccess_smap_dataset(m, d)
        
    #for name in MODIS_MCD_FILE_URLS[:1]:
    #    d = ingest_modis(name)
    #    proccess_modis_mcd_dataset(m, d, name)
    
#main()

def download_modis_mcd_datasets():
    for name in MODIS_MCD_FILE_URLS:
        MODIS_URL = MODIS_DATASET_URL + name + MODIS_VARIABLES
        MCD_URL = MCD_DATASET_URL + name + ".nc"
        print("starting dataset " + name)
        print("starting modis @ " + MODIS_URL)
        modis_file = get_file(MODIS_URL, name, "MODIS")
        #print("starting mcd @ " + MCD_URL)
        #mcd_file = get_file(MCD_URL, name, "MCD")
        print("downloaded")
    print("done")
        
download_modis_mcd_datasets()



