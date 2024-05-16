import netCDF4

dataset_url = ("http://n5eil01u.ecs.nsidc.org:80/opendap/SMAP/SPL3SMP.003/"
               "2015.05.20/SMAP_L3_SM_P_20150520_R13080_001.h5")

dataset = netCDF4.Dataset(dataset_url)
for var in dataset.variables:
    print(var)

lat = dataset.variables["Soil_Moisture_Retrieval_Data_latitude"][:,:]
lon = dataset.variables["Soil_Moisture_Retrieval_Data_longitude"][:,:]
mos = dataset.variables["Soil_Moisture_Retrieval_Data_soil_moisture"][:,:]
mos

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def show_vars(dataset):
    for var in dataset.variables:
        print(var)

plt.figure(figsize=(10,10))
b = Basemap(projection='ortho',lon_0=40,lat_0=40,resolution='l')
b.drawcoastlines()
cs = b.pcolor(lon, lat, mos, latlon=True)
cbar = b.colorbar(cs, location='bottom', pad='5%')
cbar.set_label("cm^3/cm^3")

lp_dataset_url = ("http://opendap.cr.usgs.gov/opendap/hyrax/"
               "MCD12C1.051/MCD12C1.051.ncml")
lp_dataset = netCDF4.Dataset(lp_dataset_url)
show_vars(lp_dataset)

asdc_dataset_url = ("http://l0dup05.larc.nasa.gov/opendap/MOPITT/MOP02J.005"
                    "/2002.03.02/MOP02J-20020302-L2V10.1.3.beta.hdf")
asdc_dataset = netCDF4.Dataset(asdc_dataset_url)
show_vars(asdc_dataset)

ghrc_dataset_url = ("https://ghrc.nsstc.nasa.gov:443/opendap/ssmi/f13/weekly/data/"
                    "2005/f13_ssmi_20050212v7_wk.nc")
ghrc_dataset = netCDF4.Dataset(ghrc_dataset_url)
show_vars(ghrc_dataset)

ghrc_wvc = ghrc_dataset["atmosphere_water_vapor_content"][:,:]
ghrc_lats = ghrc_dataset["latitude"][:]
ghrc_lons = ghrc_dataset["longitude"][:]
conv_lats, conv_lons = np.meshgrid(ghrc_lons, ghrc_lats)

plt.figure(figsize=(20,20))
m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawcoastlines()
m.pcolormesh(conv_lats, conv_lons, ghrc_wvc, latlon=True)

ornl_dataset_url = ("http://thredds.daac.ornl.gov/thredds/"
                    "dodsC/ornldaac/720/a785mfd.nc4")
ornl_dataset = netCDF4.Dataset(ornl_dataset_url)
show_vars(ornl_dataset)

po_dataset_url = ("http://opendap.jpl.nasa.gov:80/opendap/SeaIce/"
                  "nscat/L17/v2/S19/S1702054.HDF.Z")
po_dataset = netCDF4.Dataset(po_dataset_url)
show_vars(po_dataset)

