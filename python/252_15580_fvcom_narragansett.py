from pylab import *
get_ipython().magic('matplotlib inline')
import matplotlib.tri as Tri
import netCDF4
import datetime as dt

# DAP Data URL
# MassBay GRID for Boston Harbor, Cape Ann
url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc'
# GOM3 GRID (for Naragansett Bay, Woods Hole)
# url='http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'
# Open DAP
nc = netCDF4.Dataset(url).variables
nc.keys()

# take a look at the "metadata" for the variable "u"
print nc['u']

shape(nc['temp'])

shape(nc['nv'])

# Desired time for snapshot
# ....right now (or some number of hours from now) ...
#start = dt.datetime.utcnow() + dt.timedelta(hours=18)
# ... or specific time (UTC)
start = dt.datetime(2015,7,25,13,0,0)+dt.timedelta(hours=0-5)

# Get desired time step  
time_var = nc['time']
itime = netCDF4.date2index(start,time_var,select='nearest')

# Get lon,lat coordinates for nodes (depth)
lat = nc['lat'][:]
lon = nc['lon'][:]
# Get lon,lat coordinates for cell centers (depth)
latc = nc['latc'][:]
lonc = nc['lonc'][:]
# Get Connectivity array
nv = nc['nv'][:].T - 1 
# Get depth
h = nc['h'][:]  # depth 

dtime = netCDF4.num2date(time_var[itime],time_var.units)
daystr = dtime.strftime('%Y-%b-%d %H:%M')
print daystr

tri = Tri.Triangulation(lon,lat, triangles=nv)

# get current at layer [0 = surface, -1 = bottom]
ilayer = 0
u = nc['u'][itime, ilayer, :]
v = nc['v'][itime, ilayer, :]

#woods hole
levels=arange(-30,2,1)
ax = [-70.7, -70.6, 41.48, 41.55]
maxvel = 1.0
subsample = 2

#boston harbor
levels=arange(-34,2,1)   # depth contours to plot
ax= [-70.97, -70.82, 42.25, 42.35] # 
maxvel = 0.5
subsample = 3

#entrance Narragansett Bay
levels=arange(-34,2,1)   # depth contours to plot
ax= [-71.48, -71.32, 41.42, 41.52] # 
maxvel = 0.5
subsample = 1

# Cape Ann
levels=arange(-34,2,1)   # depth contours to plot
ax= [-70.7, -70.58, 42.575, 42.695]
maxvel = 1.
subsample = 1

# find velocity points in bounding box
ind = argwhere((lonc >= ax[0]) & (lonc <= ax[1]) & (latc >= ax[2]) & (latc <= ax[3]))

np.random.shuffle(ind)
Nvec = int(len(ind) / subsample)
idv = ind[:Nvec]

# tricontourf plot of water depth with vectors on top
figure(figsize=(18,10))
subplot(111,aspect=(1.0/cos(mean(lat)*pi/180.0)))
tricontourf(tri, -h,levels=levels,shading='faceted',cmap=plt.cm.gist_earth)
axis(ax)
gca().patch.set_facecolor('0.5')
cbar=colorbar()
cbar.set_label('Water Depth (m)', rotation=-90)
Q = quiver(lonc[idv],latc[idv],2.24*u[idv],2.24*v[idv],scale=20)
maxstr='%3.1f mph' % maxvel
qk = quiverkey(Q,0.92,0.08,maxvel,maxstr,labelpos='W')
title('NECOFS Velocity, Layer %d, %s UTC' % (ilayer, daystr));
figname = r'NECOFS_Velocity'+daystr+'.png'
print figname
savefig('NECOFS_CapeAnn_example.png')

v

