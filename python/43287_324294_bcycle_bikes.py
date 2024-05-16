import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns

from bcycle_lib.utils import *

get_ipython().magic('matplotlib inline')
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

stations_df = load_stations()
bikes_df = load_bikes()
bikes_df.head()

total_bikes_df = bikes_df.copy()
total_bikes_df = total_bikes_df.groupby('datetime').sum().reset_index()
total_bikes_df.index = total_bikes_df['datetime']
total_bikes_df = total_bikes_df.drop(['station_id', 'datetime', 'docks'], axis=1)

resampled_bikes_df = total_bikes_df.resample('3H').mean()
mean_bikes = resampled_bikes_df['bikes'].mean()
min_bikes = resampled_bikes_df['bikes'].min()
print('Mean bikes in BCycle stations: {:.0f}, minimum: {:.0f}'.format(mean_bikes, min_bikes))

xtick = pd.date_range( start=resampled_bikes_df.index.min( ), end=resampled_bikes_df.index.max( ), freq='W' )

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = resampled_bikes_df.plot(ax=ax, legend=None)
ax.axhline(y = mean_bikes, color = 'black', linestyle='dashed')
ax.set_xticks( xtick )
ax.set_ylim(ymin=200)
ax.set_xlabel('Date', fontdict={'size' : 14})
ax.set_ylabel('Bikes docked in BCycle stations', fontdict={'size' : 14})
ax.set_title('Austin BCycle Bikes stored in stations in April and May 2016', fontdict={'size' : 18, 'weight' : 'bold'})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)


# Sort the bikes_df dataframe by station_id first, and then datetime so we
# can use a diff() and get the changes by time for each station
bikes_df = load_bikes()
bikes_df = bikes_df.sort_values(['station_id', 'datetime']).copy()
stations = bikes_df['station_id'].unique()

# Our dataframe is grouped by station_id first now, so grab each station in
# turn and do a diff() on bikes and docks for each station individually
diff_list = list()
for station in stations:
    station_diff_df = bikes_df[bikes_df['station_id'] == station].copy()
    station_diff_df['bikes_diff'] = station_diff_df['bikes'].diff()
    station_diff_df['docks_diff'] = station_diff_df['docks'].diff()
    diff_list.append(station_diff_df)

# Concatenate the station dataframes back together into a single one.
# Make sure we didn't lose any rows in the process (!)
bikes_diff_df = pd.concat(diff_list)

# The first row of each station-wise diff is filled with NaNs, store a 0 in these fields
# then we can convert the data type from floats to int8s 
bikes_diff_df.fillna(0, inplace=True)
bikes_diff_df[['bikes_diff', 'docks_diff']] = bikes_diff_df[['bikes_diff', 'docks_diff']].astype(np.int8)
bikes_diff_df.index = bikes_diff_df['datetime']
bikes_diff_df.drop('datetime', axis=1, inplace=True)
assert(bikes_df.shape[0] == bikes_diff_df.shape[0]) 
bikes_diff_df.describe()

bike_trips_df = bikes_diff_df.copy()

# Checkouts are all negative `bikes_diff` values. Filter these and take abs()
bike_trips_df['checkouts'] = bike_trips_df['bikes_diff']
bike_trips_df.loc[bike_trips_df['checkouts'] > 0, 'checkouts'] = 0
bike_trips_df['checkouts'] = bike_trips_df['checkouts'].abs()

# Conversely, checkins are positive `bikes_diff` values
bike_trips_df['checkins'] = bike_trips_df['bikes_diff']
bike_trips_df.loc[bike_trips_df['checkins'] < 0, 'checkins'] = 0
bike_trips_df['checkins'] = bike_trips_df['checkins'].abs()

# Might want to use sum of checkouts and checkins for find "busiest" stations
bike_trips_df['totals'] = bike_trips_df['checkouts'] + bike_trips_df['checkins']
bike_trips_df.head()

daily_bikes_df = bike_trips_df.copy()
daily_bikes_df = daily_bikes_df.reset_index()
daily_bikes_df = daily_bikes_df[['datetime', 'station_id', 'checkouts']]
daily_bikes_df = daily_bikes_df.groupby('datetime').sum()
daily_bikes_df = daily_bikes_df.resample('1D').sum()
daily_bikes_df['weekend'] = np.where(daily_bikes_df.index.dayofweek > 4, True, False)
daily_bikes_df['color'] = np.where(daily_bikes_df['weekend'], '#467821', '#348ABD')

median_weekday = daily_bikes_df.loc[daily_bikes_df['weekend'] == False, 'checkouts'].median()
median_weekend = daily_bikes_df.loc[daily_bikes_df['weekend'] == True, 'checkouts'].median()

print('Median weekday checkouts: {:.0f}, median weekend checkouts: {:.0f}'.format(median_weekday, median_weekend))

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = daily_bikes_df['checkouts'].plot.bar(ax=ax, legend=None, color=daily_bikes_df['color'])
ax.set_xticklabels(daily_bikes_df.index.strftime('%a %b %d'))

ax.set_xlabel('', fontdict={'size' : 14})
ax.set_ylabel('Daily checkouts', fontdict={'size' : 14})
ax.set_title('Austin BCycle checkouts by day in April and May 2016', fontdict={'size' : 16})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

boxplot_trips_df = daily_bikes_df.copy()
boxplot_trips_df = boxplot_trips_df.reset_index()
boxplot_trips_df['weekday_name'] = boxplot_trips_df['datetime'].dt.weekday_name
boxplot_trips_df = boxplot_trips_df[['weekday_name', 'checkouts']]

day_names=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig, ax = plt.subplots(1,1, figsize=(16,10))  
ax = sns.boxplot(data=boxplot_trips_df, x="weekday_name", y="checkouts", order=day_names)
ax.set_xlabel('', fontdict={'size' : 14})
ax.set_ylabel('Daily checkouts', fontdict={'size' : 14})
ax.set_title('Daily checkouts', fontdict={'size' : 18})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

checkouts_df = bike_trips_df.copy()
checkouts_df = checkouts_df.reset_index()
checkouts_df['dayofweek'] = checkouts_df['datetime'].dt.weekday_name
checkouts_df['hour'] = checkouts_df['datetime'].dt.hour
checkouts_df = checkouts_df.groupby(['dayofweek', 'hour']).sum().reset_index()
checkouts_df = checkouts_df[['dayofweek', 'hour', 'checkouts']]
checkouts_df = checkouts_df.pivot_table(values='checkouts', index='hour', columns='dayofweek')

checkouts_df = checkouts_df[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
day_palette = sns.color_palette("hls", 7) # Need to have 7 distinct colours

fig, ax = plt.subplots(1,1, figsize=(16,10))
ax = checkouts_df.plot.line(ax=ax, linewidth=3, color=day_palette)
ax.set_xlabel('Hour (24H clock)', fontdict={'size' : 14})
ax.set_ylabel('Number of hourly checkouts', fontdict={'size' : 14})
ax.set_title('Hourly checkouts by day and hour in Austin BCycle stations in April and May 2016'
             ,fontdict={'size' : 18})
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.xaxis.set_ticks(checkouts_df.index)
ax.legend(fontsize=14)

heatmap_df = bike_trips_df.copy()

heatmap_df = heatmap_df.reset_index()
heatmap_df['dayofweek'] = heatmap_df['datetime'].dt.dayofweek
heatmap_df['hour'] = heatmap_df['datetime'].dt.hour
heatmap_df['weekday'] = heatmap_df['datetime'].dt.dayofweek < 5
heatmap_df = heatmap_df.groupby(['weekday', 'station_id', 'hour']).sum().reset_index()
heatmap_df = heatmap_df[['weekday', 'station_id', 'hour', 'checkouts']]
heatmap_df = heatmap_df[heatmap_df['station_id'] < 49]

heatmap_df = pd.merge(heatmap_df, stations_df[['station_id', 'name']])

weekday_df = heatmap_df[heatmap_df['weekday']].pivot_table(values='checkouts', index='name', columns='hour')
weekend_df = heatmap_df[~heatmap_df['weekday']].pivot_table(values='checkouts', index='name', columns='hour')

weekday_df = weekday_df / 5.0 # Normalize checkouts by amount of days
weekend_df = weekend_df / 2.0

weekday_max = weekday_df.max().max() # Take max over stations and hours
weekend_max = weekend_df.max().max() # Take max over stations and hours

fig, ax = plt.subplots(1, 2, figsize=(12,20))
sns.heatmap(data=weekday_df, robust=True, ax=ax[0], linewidth=2, square=True, vmin=0, vmax=weekday_max, cbar=False, cmap='Blues')
ax[0].set_xlabel('Hour of day')
ax[0].set_ylabel('')
ax[0].set_title('Weekday checkouts by station and time', fontdict={'size' : 15})
ax[0].tick_params(axis='x', labelsize=13)
ax[0].tick_params(axis='y', labelsize=13)

sns.heatmap(data=weekend_df, robust=True, ax=ax[1], linewidth=2, square=True, vmin=0, vmax=weekend_max, cbar=False, cmap='Blues', yticklabels=False)
ax[1].set_xlabel('Hour of day')
ax[1].set_ylabel('')
ax[1].set_title('Weekend checkouts by station and time', fontdict={'size' : 15})
ax[1].tick_params(axis='x', labelsize=13)
ax[1].tick_params(axis='y', labelsize=13)

# Initial setup for the visualization
map_df = bike_trips_df.copy()

map_df = map_df.reset_index()
map_df['dayofweek'] = map_df['datetime'].dt.dayofweek
map_df['hour'] = map_df['datetime'].dt.hour
map_df['3H'] = (map_df['hour'] // 3) * 3
map_df['weekday'] = map_df['datetime'].dt.dayofweek < 5

map_df = map_df.groupby(['weekday', 'station_id', '3H']).sum().reset_index()
map_df = map_df[['weekday', 'station_id', '3H', 'checkouts']]
map_df = map_df[map_df['station_id'] < 49] # Stations 49 and 50 were only open part of the time

map_df.loc[map_df['weekday'] == False, 'checkouts'] = map_df.loc[map_df['weekday'] == False, 'checkouts'] / 2.0
map_df.loc[map_df['weekday'] == True, 'checkouts'] = map_df.loc[map_df['weekday'] == True, 'checkouts'] / 5.0


map_df = pd.merge(map_df, stations_df[['station_id', 'name', 'lat', 'lon']])

# Calculate where the map should be centred based on station locations
min_lat = stations_df['lat'].min()
max_lat = stations_df['lat'].max()
min_lon = stations_df['lon'].min()
max_lon = stations_df['lon'].max()
center_lat = min_lat + (max_lat - min_lat) / 2.0
center_lon = min_lon + (max_lon - min_lon) / 2.0

# map_df.head(10)

from tqdm import tqdm

# Plot the resulting data on a map
# Hand-tuned parameter to control circle size
K = 3
C = 2

hours = range(0, 24, 3)

for weekday in (False, True):
    if weekday:
        days = 'weekdays'
    else:
        days = 'weekends'
        
    for hour in tqdm(hours, desc='Generating maps for {}'.format(days)):
        hour_df = map_df[(map_df['weekday'] == weekday) & (map_df['3H'] == hour)]

        map = folium.Map(location=(center_lat, center_lon), 
                     zoom_start=14, 
                     tiles='Stamen Toner',
                     control_scale=True)

        for station in hour_df.iterrows():
            stat = station[1]
            folium.CircleMarker([stat['lat'], stat['lon']], radius=(stat['checkouts'] * K) + C,
                                popup='{} - {} checkouts @ {}:00 - {}:00'.format(stat['name'], stat['checkouts'], stat['3H'], stat['3H']+3),
                                fill_color='blue', fill_opacity=0.8
                               ).add_to(map)

        if weekday:
            filename='weekday_{}.html'.format(hour)
        else:
            filename='weekend_{}.html'.format(hour)

        map.save(filename)
    
print('Completed map HTML generation!')

