import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns

import datetime

from bcycle_lib.utils import *

get_ipython().magic('matplotlib inline')
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

weather_df = load_weather()
weather_df.head(6)

weather_df.describe()

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = weather_df.plot(y=['max_temp', 'min_temp'], ax=ax)
ax.legend(fontsize=13)
xtick = pd.date_range( start=weather_df.index.min( ), end=weather_df.index.max( ), freq='D' )
ax.set_xticks( xtick )
# ax.set_xticklabels(weather_df.index.strftime('%a %b %d'))
ax.set_xlabel('Date', fontdict={'size' : 14})
ax.set_ylabel('Temperature (°F)', fontdict={'size' : 14})
ax.set_title('Austin Minimum and Maximum Temperatures during April and May 2016', fontdict={'size' : 16}) 
# fig.autofmt_xdate(rotation=90)
ttl = ax.title
ttl.set_position([.5, 1.02])
ax.legend(['Max Temp', 'Min Temp'], fontsize=14, loc=1)



ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

fig, ax = plt.subplots(1,2, figsize=(12,6))

# ax[0] = weather_df['min_temp'].plot.hist(ax=ax[0]) # sns.distplot(weather_df['min_temp'], ax=ax[0])
# ax[1] = weather_df['max_temp'].plot.hist(ax=ax[1]) # sns.distplot(weather_df['max_temp'], ax=ax[1])

ax[0] = sns.distplot(weather_df['min_temp'], ax=ax[0])
ax[1] = sns.distplot(weather_df['max_temp'], ax=ax[1])

for axis in ax:
    axis.set_xlabel('Temperature (°F)', fontdict={'size' : 14})
    axis.set_ylabel('Density', fontdict={'size' : 14})

ax[0].set_title('Minimum Temperature Distribution', fontdict={'size' : 16}) 
ax[1].set_title('Maximum Temperature Distribution', fontdict={'size' : 16}) 

g = sns.pairplot(data=weather_df[['min_temp', 'max_temp']], kind='reg',size=4)

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = weather_df.plot(y=['max_pressure', 'min_pressure'], ax=ax)
ax.legend(fontsize=13)
xtick = pd.date_range( start=weather_df.index.min( ), end=weather_df.index.max( ), freq='D' )
ax.set_xticks( xtick )
# ax.set_xticklabels(weather_df.index.strftime('%a %b %d'))
ax.set_xlabel('Date', fontdict={'size' : 14})
ax.set_ylabel('Pressure (inches)', fontdict={'size' : 14})
ax.set_title('Min and Max Pressure', fontdict={'size' : 18}) 
# fig.autofmt_xdate(rotation=90)

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = weather_df['precipitation'].plot.bar(ax=ax, legend=None)
ax.set_xticklabels(weather_df.index.strftime('%a %b %d'))
ax.set_xlabel('', fontdict={'size' : 14})
ax.set_ylabel('Precipitation (inches)', fontdict={'size' : 14})
ax.set_title('Austin Precipitation in April and May 2016', fontdict={'size' : 16})
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=14)
ttl = ax.title
ttl.set_position([.5, 1.02])

fig, ax = plt.subplots(1,1, figsize=(6,6))
ax = weather_df['precipitation'].plot.hist(ax=ax)
ax.set_xlabel('Precipitation (inches)', fontdict={'size' : 14})
ax.set_ylabel('Count', fontdict={'size' : 14})
ax.set_title('Precipitation distribution', fontdict={'size' : 16}) 

fig, ax = plt.subplots(1,1, figsize=(18,10))
ax = weather_df.plot(y=['max_wind', 'min_wind', 'max_gust'], ax=ax)
ax.legend(fontsize=13)
xtick = pd.date_range( start=weather_df.index.min( ), end=weather_df.index.max( ), freq='D' )
ax.set_xticks( xtick )
# ax.set_xticklabels(weather_df.index.strftime('%a %b %d'))
ax.set_xlabel('Date', fontdict={'size' : 14})
ax.set_ylabel('Wind speed (MPH)', fontdict={'size' : 14})
ax.set_title('Wind speeds', fontdict={'size' : 18}) 
# fig.autofmt_xdate(rotation=90)

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

g = sns.pairplot(data=weather_df[['min_wind', 'max_wind', 'max_gust']], kind='reg',size=3.5)

# weather_df[['thunderstorm', 'rain', 'fog']].plot.bar(figsize=(20,20))
heatmap_df = weather_df.copy()
heatmap_df = heatmap_df[['thunderstorm', 'rain', 'fog']]
heatmap_df = heatmap_df.reset_index()
heatmap_df['day'] = heatmap_df['date'].dt.dayofweek
heatmap_df['week'] = heatmap_df['date'].dt.week
heatmap_df = heatmap_df.pivot_table(values='thunderstorm', index='day', columns='week')
heatmap_df = heatmap_df.fillna(False)
# ['day'] = heatmap_df.index.dt.dayofweek

# Restore proper day and week-of-month labels. 
heatmap_df.index = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
weeks = heatmap_df.columns
weeks = ['2016-W' + str(week) for week in weeks] # Convert to '2016-Wxx'
weeks = [datetime.datetime.strptime(d + '-0', "%Y-W%W-%w").strftime('%b %d') for d in weeks]
heatmap_df.columns = weeks

fig, ax = plt.subplots(1,1, figsize=(8, 6))
sns.heatmap(data=heatmap_df, square=True, cmap='Blues', linewidth=2, cbar=False, linecolor='white', ax=ax)
ax.set_title('Thunderstorms by day and week', fontdict={'size' : 18})
ttl = ax.title
ttl.set_position([.5, 1.05])
ax.set_xlabel('Week ending (Sunday)', fontdict={'size' : 14})
ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)

