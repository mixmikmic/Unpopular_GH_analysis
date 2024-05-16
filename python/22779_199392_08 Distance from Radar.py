get_ipython().magic('matplotlib inline')
import os
import numpy as np
import pandas as pd
from pointprocess.region import Region

import matplotlib.pyplot as plt
from geopy.distance import vincenty, great_circle

# this is where my data path and colormap are set
from lightning_setup import *

def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    
    https://gist.github.com/jeromer/2005586
    """
    import math
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

get_ipython().run_cell_magic('time', '', "bearing = [calculate_initial_compass_bearing(c.CENTER, (lat, lon)) for \n           lat, lon in lat_lon[['lat','lon']].values]\n\nmin_bearing = min(bearing)\nmax_bearing = max(bearing)\nprint(min_bearing, max_bearing)")

lat_lon.plot.scatter('lon', 'lat')
plt.scatter(c.CENTER[1], c.CENTER[0], c='r')

import xarray as xr

get_ipython().run_cell_magic('time', '', 'for m in range(4,10):\n    dist = []\n    ds = xr.open_mfdataset(c.PATH+\'{y}_{m:02d}_*.nc\'.format(y=\'201*\', m=m))\n                                                            \n    # this line needs to be changed to reflect the city\n    if city == \'greer\':\n        ds0 = ds.where((ds.lon>c.CENTER[1]) & (ds.lat>c.CENTER[0]) & (ds.cloud_ground == b\'G\')).dropna(\'record\')\n    elif city == \'stlouis\':\n        ds0 = ds.where((ds.lon>c.CENTER[1]) & (ds.cloud_ground == b\'G\')).dropna(\'record\')\n    elif city == \'philly\':\n        ds0 = ds.where((ds.lon<c.CENTER[1]) & (ds.cloud_ground == b\'G\')).dropna(\'record\')\n    elif city == \'kansas\':\n        ds0 = ds.where((ds.lon<c.CENTER[1]) & (ds.lat>c.CENTER[0]) & (ds.cloud_ground == b\'G\')).dropna(\'record\')\n    else:\n        ds0 = ds\n\n    print(ds0.record.shape)\n    locs = np.stack([ds0.lat.values, ds0.lon.values], axis=1)\n    ds0.close()\n    ds.close()\n    for lat, lon in locs:\n        bearing = calculate_initial_compass_bearing(c.CENTER, (lat, lon))\n        if min_bearing < bearing < max_bearing:                                                    \n            dist.append(vincenty(c.CENTER, (lat, lon)).kilometers)   \n    computed = pd.HDFStore("computed")\n    computed[\'dist_from_radar_2010_2015_{mm:02d}_{city}_city_area\'.format(mm=m, city=city)] = pd.Series(dist)\n    computed.close()\n    print(m)')

get_ipython().run_cell_magic('time', '', 'hist = {}\nfor m in range(4,10):\n    computed = pd.HDFStore("computed")\n    dist = computed[\'dist_from_radar_2010_2015_{mm:02d}_{city}_city_area\'.format(mm=m, city=city)].values\n    computed.close()\n    FC, edges = np.histogram(dist, range(0, 220, 1))\n    area = [np.pi*(edges[i+1]**2-edges[i]**2)*(max_bearing-min_bearing)/360. for i in range(len(FC))]\n    hist.update({m: FC/area/25})\ncenters = (edges[1:]+edges[:-1])/2.\nhist.update({\'km_from_radar\': centers})\ndf = pd.DataFrame.from_dict(hist).set_index(\'km_from_radar\')\ndf.columns.name = \'mean_annual_flash_density\'')

n=1
plt.figure(figsize=(10,10))
for m in range(4,10):
    ax = plt.subplot(3,2,n)
    ax.plot(centers, hist[m])
    if n%2 == 1:
        ax.set_ylabel("Flash Density [strikes/km/year]")
    if n>4:
        ax.set_xlabel("Distance from radar [km]")
    ax.set_xlim(30,210);
    ax.set_ylim(0,.4)
    ax.set_title(months[m])
    n+=1

df.to_csv("Kansas City monthly CG FD as a function of distance from radar.csv")

