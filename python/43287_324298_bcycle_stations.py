import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns

from bcycle_lib.utils import *

get_ipython().magic('matplotlib inline')

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Load the stations table, and show the first 10 entries
STATIONS = 5
stations_df = load_stations()
num_stations = stations_df.shape[0]
print('Found {} stations, showing first {}'.format(num_stations, STATIONS))
stations_df.head(STATIONS)

# Calculate where the map should be centred based on station locations
min_lat = stations_df['lat'].min()
max_lat = stations_df['lat'].max()
min_lon = stations_df['lon'].min()
max_lon = stations_df['lon'].max()
center_lat = min_lat + (max_lat - min_lat) / 2.0
center_lon = min_lon + (max_lon - min_lon) / 2.0

# Plot map using the B&W Stamen Toner tiles centred on BCycle stations
map = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='Stamen Toner',
                 control_scale=True)

# Add markers to the map for each station. Click on them to see their name
for station in stations_df.iterrows():
    stat=station[1]
    folium.Marker([stat['lat'], stat['lon']],
              popup=stat['name'],
              icon=folium.Icon(icon='info-sign')
             ).add_to(map)

map.save('stations.html')
map

# Load bikes dataframe, calculate the capacity of each every 5 minutes (bikes + docks)
bikes_df = load_bikes()
bikes_df['capacity'] = bikes_df['bikes'] + bikes_df['docks']

# Now find the max capacity across all the stations at all 5 minute intervals
bikes_df = bikes_df.groupby('station_id').max().reset_index()
bikes_df = bikes_df[['station_id', 'capacity']]

# Now join with the stations dataframe using station_id
stations_cap_df = pd.merge(stations_df, bikes_df, on='station_id')

# Print the smallest and largest stations
N = 4
sorted_stations = stations_cap_df.sort_values(by='capacity', ascending=True)
print('Smallest {} stations: \n{}\n'.format(N, sorted_stations[['name', 'capacity']][:N]))
print('Largest {} stations: \n{}\n'.format(N, sorted_stations[['name', 'capacity']][-N:]))

# Show a histogram of the capacities
# fig = plt.figure()

ax1 = stations_cap_df['capacity'].plot.hist(figsize=(10,6))
ax1.set_xlabel('Station Capacity', fontsize=14)
ax1.set_ylabel('Number of stations', fontsize=14)
ax1.set_title('Histogram of station capacities', fontsize=14)

# Now plot each station as a circle whose area represents the capacity
map = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='Stamen Toner',
                 control_scale=True)

# Hand-tuned values to make differences between circles larger
K = 0.5 
P = 2

# Add markers whose radius is proportional to station capacity. 
# Click on them to pop up their name and capacity
for station in stations_cap_df.iterrows():
    stat=station[1]
    folium.CircleMarker([stat['lat'], stat['lon']],
                        radius= K * (stat['capacity'] ** P), # Scale circles to show difference
                        popup='{} - capacity {}'.format(stat['name'], stat['capacity']),
                        fill_color='blue',
                        fill_opacity=0.8
                       ).add_to(map)
map.save('station_capacity.html')
map

# Load both the bikes and station dataframes
bikes_df = load_bikes()
stations_df = load_stations()

# Using the bikes and stations dataframes, mask off so the only rows remaining
# are either empty or full cases from 6AM onwards
bike_empty_mask = bikes_df['bikes'] == 0
bike_full_mask = bikes_df['docks'] == 0
bike_empty_full_mask = bike_empty_mask | bike_full_mask

bikes_empty_full_df = bikes_df[bike_empty_full_mask].copy()
bikes_empty_full_df['empty'] =  bikes_empty_full_df['bikes'] == 0
bikes_empty_full_df['full'] = bikes_empty_full_df['docks'] == 0
bikes_empty_full_df.head()

# Now aggregate the remaining rows by station_id, and plot the results
bike_health_df = bikes_empty_full_df.copy()
bike_health_df = bike_health_df[['station_id', 'empty', 'full']].groupby('station_id').sum().reset_index()
bike_health_df = pd.merge(bike_health_df, stations_df, on='station_id')
bike_health_df['oos'] = bike_health_df['full'] + bike_health_df['empty'] 
bike_health_df = bike_health_df.sort_values('oos', ascending=False)

ax1 = (bike_health_df[['name', 'empty', 'full']]
       .plot.bar(x='name', y=['empty', 'full'], stacked=True, figsize=(16,8)))
ax1.set_xlabel('Station', fontsize=14)
ax1.set_ylabel('# 5 minute periods empty or full', fontsize=14)
ax1.set_title('Empty/Full station count during April/May 2016',  fontdict={'size' : 18, 'weight' : 'bold'})
ax1.tick_params(axis='x', labelsize=13)
ax1.tick_params(axis='y', labelsize=13)
ax1.legend(fontsize=13)

# For this plot, we don't want to mask out the time intervals where stations are neither full nor empty.
HEALTHY_RATIO = 0.9
station_ratio_df = bikes_df.copy()
station_ratio_df['empty'] = station_ratio_df['bikes'] == 0
station_ratio_df['full'] = station_ratio_df['docks'] == 0
station_ratio_df['neither'] = (station_ratio_df['bikes'] != 0) & (station_ratio_df['docks'] != 0)

station_ratio_df = station_ratio_df[['station_id', 'empty', 'full', 'neither']].groupby('station_id').sum().reset_index()
station_ratio_df['total'] = station_ratio_df['empty'] + station_ratio_df['full'] + station_ratio_df['neither']
station_ratio_df = pd.merge(station_ratio_df, stations_df, on='station_id')

station_ratio_df['full_ratio'] = station_ratio_df['full'] / station_ratio_df['total']
station_ratio_df['empty_ratio'] = station_ratio_df['empty'] / station_ratio_df['total']
station_ratio_df['oos_ratio'] = station_ratio_df['full_ratio'] + station_ratio_df['empty_ratio']
station_ratio_df['in_service_ratio'] = 1 - station_ratio_df['oos_ratio']
station_ratio_df['healthy'] = station_ratio_df['in_service_ratio'] >= HEALTHY_RATIO
station_ratio_df['color'] = np.where(station_ratio_df['healthy'], '#348ABD', '#A60628')

station_ratio_df = station_ratio_df.sort_values('in_service_ratio', ascending=False)
colors = ['b' if ratio >= 0.9 else 'r' for ratio in station_ratio_df['in_service_ratio']]

# station_ratio_df.head()
ax1 = (station_ratio_df.sort_values('in_service_ratio', ascending=False)
       .plot.bar(x='name', y='in_service_ratio', figsize=(16,8), legend=None, yticks=np.linspace(0.0, 1.0, 11),
                color=station_ratio_df['color']))
ax1.set_xlabel('Station', fontsize=14)
ax1.set_ylabel('%age of time neither empty nor full', fontsize=14)
ax1.set_title('In-service percentage by station during April/May 2016',  fontdict={'size' : 16, 'weight' : 'bold'})
ax1.axhline(y = HEALTHY_RATIO, color = 'black')
ax1.tick_params(axis='x', labelsize=13)
ax1.tick_params(axis='y', labelsize=13)

mask = station_ratio_df['healthy'] == False
unhealthy_stations_df = station_ratio_df[mask].sort_values('oos_ratio', ascending=False)
unhealthy_stations_df = pd.merge(unhealthy_stations_df, stations_cap_df[['station_id', 'capacity']], on='station_id')
unhealthy_stations_df[['name', 'oos_ratio', 'full_ratio', 'empty_ratio', 'capacity']].reset_index(drop=True).round(2)

# Merge in the station capacity also for the popup markers
station_ratio_cap_df = pd.merge(station_ratio_df, stations_cap_df[['station_id', 'capacity']], on='station_id')

map = folium.Map(location=(center_lat, center_lon), 
                 zoom_start=14, 
                 tiles='Stamen Toner',
                 control_scale=True)

# Hand-tuned parameter to increase circle size
K = 1000
C = 5
for station in station_ratio_cap_df.iterrows():
    stat = station[1]
    
    if stat['healthy']:
        colour = 'blue'
    else:
        colour='red'
    
    folium.CircleMarker([stat['lat'], stat['lon']], radius=(stat['oos_ratio'] * K) + C,
                        popup='{}, empty {:.1f}%, full {:.1f}%, capacity {}'.format(
                          stat['name'], stat['empty_ratio']*100, stat['full_ratio']*100, stat['capacity']),
                        fill_color=colour, fill_opacity=0.8
                       ).add_to(map)

map.save('unhealthy_stations.html')
map

# Plot the empty/full time periods grouped by hour for the top 10 
oos_stations_df = bikes_df.copy()
oos_stations_df['empty'] = oos_stations_df['bikes'] == 0
oos_stations_df['full'] = oos_stations_df['docks'] == 0
oos_stations_df['neither'] = (oos_stations_df['bikes'] != 0) & (oos_stations_df['docks'] != 0)
oos_stations_df['hour'] = oos_stations_df['datetime'].dt.hour

oos_stations_df = (oos_stations_df[['station_id', 'hour', 'empty', 'full', 'neither']]
                   .groupby(['station_id', 'hour']).sum().reset_index())
oos_stations_df = oos_stations_df[oos_stations_df['station_id'].isin(unhealthy_stations_df['station_id'])]
oos_stations_df['oos'] = oos_stations_df['empty'] + oos_stations_df['full'] 
oos_stations_df = pd.merge(stations_df, oos_stations_df, on='station_id')

oos_stations_df

g = sns.factorplot(data=oos_stations_df, x="hour", y="oos", col='name',
                   kind='bar', col_wrap=2, size=3.5, aspect=2.0, color='#348ABD')

bikes_capacity_df = bikes_df.copy()
bikes_capacity_df['capacity'] = bikes_capacity_df['bikes'] + bikes_capacity_df['docks']

# Now find the max capacity across all the stations at all 5 minute intervals
bikes_capacity_df = bikes_capacity_df.groupby('station_id').max().reset_index()
bike_merged_health_df = pd.merge(bike_health_df, 
                                 bikes_capacity_df[['station_id', 'capacity']], 
                                 on='station_id', 
                                 how='inner')

plt.rc("legend", fontsize=14) 
sns.jointplot("capacity", "full", data=bike_merged_health_df, kind="reg", size=8)
plt.xlabel('Station capacity', fontsize=14)
plt.ylabel('5-minute periods that are full', fontsize=14)
plt.tick_params(axis="both", labelsize=14)


sns.jointplot("capacity", "empty", data=bike_merged_health_df, kind="reg", size=8)
plt.xlabel('Station capacity', fontsize=14)
plt.ylabel('5-minute periods that are empty', fontsize=14)
plt.tick_params(axis="both", labelsize=14)

bikes_df = load_bikes()
empty_mask = bikes_df['bikes'] == 0
full_mask = bikes_df['docks'] == 0
empty_full_mask = empty_mask | full_mask
bikes_empty_full_df = bikes_df[empty_full_mask].copy()
bikes_empty_full_df['day_of_week'] = bikes_empty_full_df['datetime'].dt.dayofweek
bikes_empty_full_df['hour'] = bikes_empty_full_df['datetime'].dt.hour

fig, axes = plt.subplots(1,2, figsize=(16,8))
bikes_empty_full_df.groupby(['day_of_week']).size().plot.bar(ax=axes[0], legend=None)
axes[0].set_xlabel('Day of week (0 = Monday, 1 = Tuesday, .. ,6 = Sunday)')
axes[0].set_ylabel('Station empty/full count per 5-minute interval ')
axes[0].set_title('Station empty/full by day of week', fontsize=15)
axes[0].tick_params(axis='x', labelsize=13)
axes[0].tick_params(axis='y', labelsize=13)

bikes_empty_full_df.groupby(['hour']).size().plot.bar(ax=axes[1])
axes[1].set_xlabel('Hour of day (24H clock)')
axes[1].set_ylabel('Station empty/full count per 5-minute interval ')
axes[1].set_title('Station empty/full by hour of day', fontsize=15)
axes[1].tick_params(axis='x', labelsize=13)
axes[1].tick_params(axis='y', labelsize=13)

