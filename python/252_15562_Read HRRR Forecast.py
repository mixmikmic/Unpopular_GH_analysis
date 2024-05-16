import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

# Resolve the latest HRRR dataset
from siphon.catalog import TDSCatalog
latest_hrrr = TDSCatalog('http://thredds-jumbo.unidata.ucar.edu/thredds/catalog/grib/HRRR/CONUS_3km/surface/latest.xml')
hrrr_ds = list(latest_hrrr.datasets.values())[0]

# Set up access via NCSS
from siphon.ncss import NCSS
ncss = NCSS(hrrr_ds.access_urls['NetcdfSubset'])

# Create a query to ask for all times in netcdf4 format for
# the Temperature_surface variable, with a bounding box
query = ncss.query()
dap_url = hrrr_ds.access_urls['OPENDAP']

query.all_times().accept('netcdf4').variables('u-component_of_wind_height_above_ground',
                                              'v-component_of_wind_height_above_ground')
query.lonlat_box(45, 41., -63, -71.5)

# Get the raw bytes and write to a file.
data = ncss.get_data_raw(query)
with open('test_uv.nc', 'wb') as outf:
    outf.write(data)

import xray
nc = xray.open_dataset('test_uv.nc')
nc

uvar_name='u-component_of_wind_height_above_ground'
vvar_name='v-component_of_wind_height_above_ground'
uvar = nc[uvar_name]
vvar = nc[vvar_name]
grid = nc[uvar.grid_mapping]
grid

uvar

lon0 = grid.longitude_of_central_meridian
lat0 = grid.latitude_of_projection_origin
lat1 = grid.standard_parallel
earth_radius = grid.earth_radius

import cartopy
import cartopy.crs as ccrs
#cartopy wants meters, not km
x = uvar.x.data*1000.
y = uvar.y.data*1000.

#globe = ccrs.Globe(ellipse='WGS84') #default
globe = ccrs.Globe(ellipse='sphere', semimajor_axis=grid.earth_radius)

crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, 
                            standard_parallels=(lat0,lat1), globe=globe)
print(uvar.x.data.shape)
print(uvar.y.data.shape)
print(uvar.time1.shape)

uvar[-1,:,:].time1.data

klev = 0
u = uvar[istep,klev,:,:].data
v = vvar[istep,klev,:,:].data
spd = np.sqrt(u*u+v*v)

fig = plt.figure(figsize=(10,16))
ax = plt.axes(projection=ccrs.PlateCarree())
c = ax.pcolormesh(x,y,spd, transform=crs,zorder=0)
cb = fig.colorbar(c,orientation='vertical',shrink=0.5)
cb.set_label('m/s')
ax.coastlines(resolution='10m',color='gray',zorder=1,linewidth=3)
ax.quiver(x,y,u,v,transform=crs,zorder=2,scale=100)
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title(uvar[istep].time1.data);
plt.axis([-71.2, -70., 42.3, 43])
#plt.axis([-72,-69.8,40.6, 43.5]);

