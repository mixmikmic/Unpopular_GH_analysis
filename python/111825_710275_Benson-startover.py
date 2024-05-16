from __future__ import print_function, division

from collections import defaultdict
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

get_ipython().magic('matplotlib inline')

path = '/Users/aleksandra/Desktop/output_data/data_mta/'

def read_data(path):
    '''opens all txt files in the given directory and creates a df'''
    all_data = pd.DataFrame()
    for filename in os.listdir(path):
        with open(path+str(filename)) as f:
            df = pd.read_csv(f)
            all_data = all_data.append(df)
    all_data.columns = ['C/A', 'UNIT', 'SCP', 'STATION', 'LINENAME', 'DIVISION', 'DATE', 'TIME','DESC', 'ENTRIES', 'EXITS']
    return all_data

def save_as_csv(df, path):
    '''saves df in given directory'''
    df.to_csv(path+'all_data.csv')

def get_data(week_nums):
    '''NOT MY CODE. downlaods all files in the list of weeks and puts them in a pandas df'''
    url = "http://web.mta.info/developers/data/nyct/turnstile/turnstile_{}.txt"
    dfs = []
    for week_num in week_nums:
        file_url = url.format(week_num)
        dfs.append(pd.read_csv(file_url))
    return pd.concat(dfs)

data = read_data(path)

data.head()

data.tail()

data['DATE_TIME'] = pd.to_datetime(data['DATE'] +' ' + data['TIME'], format = '%m/%d/%Y %H:%M:%S')

data.head()

sorting = lambda x: ''.join(sorted(x))
data['LINE_ADJUST'] = data["LINENAME"].map(sorting)
data['UN_STATION'] = data['STATION'].map(str) + data['LINE_ADJUST'].map(str)

data.head()

def repl(word):
    return re.sub('\W', '', word)

data['UN_STATION'] = data['UN_STATION'].map(repl)

data.head()

turnstile = ['C/A', 'UNIT', 'SCP', 'STATION', 'UN_STATION']

data.groupby(turnstile + ['DATE_TIME']).ENTRIES.count().sort_values(ascending = False).head()

filtre = ((data["C/A"] == "N418") & 
(data["UNIT"] == "R269") & 
(data["SCP"] == "01-05-00") & 
(data["STATION"] == "BEDFORD-NOSTRAN") &
(data["DATE_TIME"].dt.date == datetime(2016, 8, 8).date()))

data[filtre].head(20)

data_no_dupl = data[data.DESC != 'RECOVR AUD']

data_no_dupl.head()

data_no_dupl.groupby(station + ['DATE_TIME']).ENTRIES.count().sort_values(ascending = False)

# drop DESC columns, there's no useful info in there anymore
data_no_dupl = data_no_dupl.drop(['DESC'], axis = 1)

data_no_dupl.head()

data_daily_entries = data_no_dupl.groupby(turnstile+['DATE']).ENTRIES.first().reset_index()

data_daily_entries.head()

data_daily_exits = data_no_dupl.groupby(turnstile+['DATE']).EXITS.first().reset_index()

data_daily_exits.head()

data_daily_entries[["PREV_DATE", "PREV_ENTRIES"]] = (data_daily_entries
                                                       .groupby(turnstile)["DATE", "ENTRIES"]
                                                       .transform(lambda grp: grp.shift(1)))

data_daily_entries.head()

data_daily_exits["PREV_EXITS"] = data_daily_exits.groupby(turnstile)['EXITS'].transform(lambda grp: grp.shift(1))

data_daily_exits.head()

data_daily_entries[['EXITS', 'PREV_EXITS']] = data_daily_exits[['EXITS', 'PREV_EXITS']]

data_daily_entries.head()

data_daily_entries.dropna(subset=["PREV_DATE"], axis=0, inplace=True)

data_daily_entries.head()

crazy_turnstiles = data_daily_entries[data_daily_entries["ENTRIES"] < data_daily_entries["PREV_ENTRIES"]]

crazy_turnstiles.head()

# crazy turnstile days for entering
len(crazy_turnstiles)

crazy_turnstiles_exit = data_daily_entries[data_daily_entries["EXITS"] < data_daily_entries["PREV_EXITS"]]
crazy_turnstiles_exit.head()

# crazy turnstile days exit
len(crazy_turnstiles_exit)

# total number of turnstile days
len(data_daily_entries)

# number of turnstiles
len(data_daily_entries.groupby(turnstile))

# check if the crazy turnstile for exiting and entering coincide
ct_entries = crazy_turnstiles.groupby(turnstile).groups.keys()
len(ct_entries)

ct_exits = crazy_turnstiles_exit.groupby(turnstile).groups.keys()
len(ct_exits)

always_crazy = set.intersection(set(ct_entries), set(ct_exits))

len(always_crazy)

# identify crazy turnstiles:
all_crazies = list(set(list(ct_entries) + list(ct_exits)))

all_crazies

stations_with_ct = []
for ts in all_crazies:
    stations_with_ct.append(ts[3])

len(set(stations_with_ct))

# total number of stations:
len(data.STATION.unique())

data_daily = data_daily_entries
data_daily.head()

# drop the crazy turnstiles:
for ts in all_crazies:
    mask = ((data_daily["C/A"] == ts[0]) & (data_daily["UNIT"] == ts[1]) & 
            (data_daily["SCP"] == ts[2]) & (data_daily["STATION"] == ts[3]))
    data_daily.drop(data_daily[mask].index, inplace=True)

data_daily[(data_daily['C/A']=='A002') & (data_daily.SCP == '02-00-00')].head()

data_daily['SALDO_ENTRIES'] = data_daily['ENTRIES']-data_daily['PREV_ENTRIES']
data_daily['SALDO_EXITS'] = data_daily['EXITS']-data_daily['PREV_EXITS']
data_daily['TRANSITING'] = data_daily['SALDO_ENTRIES'] + data_daily['SALDO_EXITS'] 

data_daily.head()

data_daily.SALDO_ENTRIES.sort_values(ascending=False).head()

data_daily = data_daily[data_daily['SALDO_ENTRIES'] < 10000]

data_daily.to_csv('/Users/aleksandra/Desktop/output_data/Data/data_daily.csv')

data_daily.SALDO_ENTRIES.max()

list_of_stations =list(data_daily.UN_STATION.unique())

list_of_stations

stations_transit = data_daily.groupby(['UN_STATION', 'DATE'])['TRANSITING'].sum()
# stations_transit = data_daily.groupby(['STATION', 'DATE'], as_index=False)['TRANSITING'].sum()

stations_transit.head()

stations_transit = pd.DataFrame(stations_transit)

stations_transit['index'] = stations_transit.index
stations_transit.head()

stations_transit[['UN_STATION', 'DATE']] = stations_transit['index'].apply(pd.Series)

stations_transit.drop('index', 1, inplace=True)

stations_transit.head()

stations_transit['DATE'] = pd.to_datetime(stations_transit['DATE'])
stations_transit['DAY_OF_WEEK'] = stations_transit['DATE'].dt.weekday_name

stations_transit.head()

stations_transit.reset_index(inplace=True, drop=True)

stations_transit.head()

stations_transit.to_csv('/Users/aleksandra/Desktop/output_data/Data/per_station.csv')

transit_dw_stations = stations_transit.groupby(['UN_STATION', 'DAY_OF_WEEK'])['TRANSITING'].sum()

transit_dw_stations = pd.DataFrame(transit_dw_stations)

transit_dw_stations.head()

transit_dw_stations['index'] = transit_dw_stations.index
transit_dw_stations.head()

transit_dw_stations[['UN_STATION', 'DAY_OF_WEEK']] = transit_dw_stations['index'].apply(pd.Series)

transit_dw_stations.head()

transit_dw_stations.drop('index', 1, inplace=True)

transit_dw_stations.reset_index(inplace=True, drop=True)

transit_dw_stations.head(7)

transit_dw_stations.to_csv('/Users/aleksandra/Desktop/output_data/Data/station_dow.csv')

len(data_daily)

weather_data = pd.read_csv('/Users/aleksandra/Desktop/output_data/Data/NYC_rainy_days.csv')

weather_data.head()

data_daily.head()

data_daily['DATE'] = pd.to_datetime(data_daily['DATE'], format = '%m/%d/%Y')
data_daily.head()

small_weather = weather_data[['DATE', 'PRCP']]
small_weather['DATE'] = pd.to_datetime(small_weather['DATE'], format = '%Y/%m/%d')
small_weather.head()

data_daily_weather = pd.merge(data_daily, small_weather, on='DATE')
data_daily_weather.head()

data_daily_weather.head()

data_daily_weather.to_csv('/Users/aleksandra/Desktop/output_data/Data/data_daily_weather.csv')



