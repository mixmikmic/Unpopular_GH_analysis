get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap


def within_bbox(bbox, loc):
    """Determine whether given location is within given bounding box.
    
    Bounding box is a dict with ll_lon, ll_lat, ur_lon and ur_lat keys
    that locate the lower left and upper right corners.
    
    The loction argument is a tuple of longitude and latitude values.
    """
    
    return bbox['ll_lon'] < loc[0] < bbox['ur_lon'] and bbox['ll_lat'] < loc[1] < bbox['ur_lat']

df = pd.read_csv('csv/POIWorld-pub.csv')
df.dropna(axis=1, how='any', inplace=True)
print(len(df))
df.head()

df.amenity.value_counts().head()

df = df[df.amenity.str.match(r'\bpub\b')]
df.amenity.value_counts()

bbox = {
    'lon': -5.23636,
    'lat': 53.866772,
    'll_lon': -10.65073,
    'll_lat': 49.16209,
    'ur_lon': 1.76334,
    'ur_lat': 60.860699
}

locations = [loc for loc in zip(df.Longitude.values, df.Latitude.values) if within_bbox(bbox, loc)]
len(locations)

fig = plt.figure(figsize=(20, 30))
markersize = 1
markertype = ','  # pixel
markercolor = '#325CA9'  # blue
markeralpha = .8 #  a bit of transparency

m = Basemap(
    projection='mill', lon_0=bbox['lon'], lat_0=bbox['lat'],
    llcrnrlon=bbox['ll_lon'], llcrnrlat=bbox['ll_lat'],
    urcrnrlon=bbox['ur_lon'], urcrnrlat=bbox['ur_lat'])

# Avoid border around map.
m.drawmapboundary(fill_color='#ffffff', linewidth=.0)

# Convert locations to x/y coordinates and plot them as dots.
lons, lats = zip(*locations)
x, y = m(lons, lats)
m.scatter(x, y, markersize, marker=markertype, color=markercolor, alpha=markeralpha)

# Set the map title.
plt.annotate('Britain & Ireland\ndrawn from pubs',
             xy=(0, .87),
             size=120, 
             xycoords='axes fraction',
             color='#888888', 
             family='Gloria')

# Set the map footer.
plt.annotate('Author: Ramiro Gómez - ramiro.org • Data: OpenStreetMap - openstreetmap.org', 
             xy=(0, 0), 
             size=14, 
             xycoords='axes fraction',
             color='#666666',
             family='Droid Sans')

plt.savefig('img/britain-ireland-drawn-from-pubs.png', bbox_inches='tight')

get_ipython().magic('signature')

