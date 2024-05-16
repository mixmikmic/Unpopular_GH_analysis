from __future__ import print_function, division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb

get_ipython().magic('matplotlib inline')

path = '/Users/aleksandra/Desktop/output_data/Data/data_daily_weather.csv'
with open(path) as f:
    weather = pd.read_csv(f)

weather.head()

df = weather[['STATION', 'UN_STATION', 'DATE', 'TRANSITING']]
df.head()

transit_station = df.groupby('UN_STATION', as_index=False).sum()
df_busiest_stations = transit_station.sort_values('TRANSITING', ascending=False).head(20)

df_busiest_stations.head()

busiest_stations = df_busiest_stations.UN_STATION

busy_df = pd.DataFrame()

for station in busiest_stations:
    busy_df = busy_df.append(df.loc[df['UN_STATION'] == station])

busy_df.head()

busy_df['DATE'] = pd.to_datetime(busy_df['DATE'])

busy_df['DAY_OF_WEEK'] = busy_df['DATE'].dt.weekday_name

busy_df.head()

busy_df.tail()

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
mapping = {day: i for i, day in enumerate(weekdays)}
busy_df['WD_NUM'] = busy_df['DAY_OF_WEEK'].map(mapping)

busy_df.head()

per_day = busy_df.groupby('WD_NUM', as_index=True).sum()

per_day['DAY_OF_WEEK'] = weekdays

per_day

per_day.TRANSITING.max()

plt.figure(figsize=(10,3))
pl = per_day.plot(x = 'DAY_OF_WEEK', y= 'TRANSITING', kind='bar'
                  , legend=False, title = 'TRANSIT PER DAY OF THE WEEK', rot=45, fontsize=10, colormap = 'ocean')
pl.set_xlabel('Weekday')
pl.set_ylabel('Transit')

fig = pl.get_figure()
fig.savefig('Data/plot_transit_dow.pdf', bbox_inches="tight")

