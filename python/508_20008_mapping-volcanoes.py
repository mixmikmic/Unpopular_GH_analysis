get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import json

from lxml import html
from mpl_toolkits.basemap import Basemap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

chartinfo = 'Author: Ramiro Gómez - ramiro.org | Data: Volcano World - volcano.oregonstate.edu'

url ='http://volcano.oregonstate.edu/oldroot/volcanoes/alpha.html'
xpath = '//table'
tree = html.parse(url)
tables = tree.xpath(xpath)

table_dfs = []
for idx in range(4, len(tables)):
    df = pd.read_html(html.tostring(tables[idx]), header=0)[0]
    table_dfs.append(df)

df_volc = pd.concat(table_dfs, ignore_index=True)

print(len(df_volc))
df_volc.head(10)

df_volc['Type'].value_counts()

def cleanup_type(s):
    if not isinstance(s, str):
        return s
    s = s.replace('?', '').replace('  ', ' ')
    s = s.replace('volcanoes', 'volcano')
    s = s.replace('volcanoe', 'Volcano')
    s = s.replace('cones', 'cone')
    s = s.replace('Calderas', 'Caldera')
    return s.strip().title()

df_volc['Type'] = df_volc['Type'].map(cleanup_type)
df_volc['Type'].value_counts()

df_volc.dropna(inplace=True)
len(df_volc)

df_volc = df_volc[df_volc['Elevation (m)'] >= 0]
len(df_volc)

def plot_map(lons, lats, elevations, projection='mill', llcrnrlat=-80, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='i', min_marker_size=2):
    bins = np.linspace(0, elevations.max(), 10)
    marker_sizes = np.digitize(elevations, bins) + min_marker_size

    m = Basemap(projection=projection, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution=resolution)
    m.drawcoastlines()
    m.drawmapboundary()
    m.fillcontinents(color = '#333333')
    
    for lon, lat, msize in zip(lons, lats, marker_sizes):
        x, y = m(lon, lat)
        m.plot(x, y, '^r', markersize=msize, alpha=.7)
    
    return m

plt.figure(figsize=(16, 8))
df = df_volc[df_volc['Type'] == 'Stratovolcano']
plot_map(df['Longitude'], df['Latitude'], df['Elevation (m)'])
plt.annotate('Stratovolcanoes of the world | ' + chartinfo, xy=(0, -1.04), xycoords='axes fraction')

plt.figure(figsize=(12, 10))
plot_map(df_volc['Longitude'], df_volc['Latitude'], df_volc['Elevation (m)'],
         llcrnrlat=5.5, urcrnrlat=83.2, llcrnrlon=-180, urcrnrlon=-52.3, min_marker_size=4)
plt.annotate('Volcanoes of North America | ' + chartinfo, xy=(0, -1.03), xycoords='axes fraction')

plt.figure(figsize=(18, 8))
plot_map(df_volc['Longitude'], df_volc['Latitude'], df_volc['Elevation (m)'],
         llcrnrlat=-11.1, urcrnrlat=6.1, llcrnrlon=95, urcrnrlon=141.1, min_marker_size=4)
plt.annotate('Volcanoes of Indonesia | ' + chartinfo, xy=(0, -1.04), xycoords='axes fraction')

plt.figure(figsize=(20, 12))
m = plot_map(df_volc['Longitude'], df_volc['Latitude'], df_volc['Elevation (m)'], min_marker_size=2)
m.warpimage(image='img/raw-bathymetry.jpg', scale=1)

plt.title('Volcanoes of the World', color='#000000', fontsize=40)
plt.annotate(chartinfo + ' | Image: NASA - nasa.gov',
             (0, 0), color='#bbbbbb', fontsize=11)
plt.show()

df_globe_values = df_volc[['Latitude', 'Longitude', 'Elevation (m)']]
globe_values = df_globe_values.as_matrix().flatten().tolist()
with open('json/globe_volcanoes.json', 'w') as f:
    json.dump(globe_values, f)

get_ipython().magic('signature')

