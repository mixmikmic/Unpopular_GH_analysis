get_ipython().magic('load_ext signature')

import os

import pandas as pd
import geonamescache

data_dir = os.path.expanduser('~/data')
gc = geonamescache.GeonamesCache()
df = pd.read_csv(os.path.join(data_dir, 'reddit', 'rplace-country-color-propensity.csv'))

df.head()

df_map = df.dropna().copy()
names = gc.get_countries()
df_map['iso3'] = df_map['iso_country_code'].apply(lambda x: names[x]['iso3'] if x in names else None)

df_map[df_map['iso3'].isnull()]

df_map.dropna(inplace=True)
df_map['top_color'] = df_map._get_numeric_data().idxmax(axis='columns').apply(lambda x: x.replace('color_', ''))

df_map[['iso3', 'top_color']].to_csv('./data/rplace-country-color-propensity.csv', index=False)

signature

